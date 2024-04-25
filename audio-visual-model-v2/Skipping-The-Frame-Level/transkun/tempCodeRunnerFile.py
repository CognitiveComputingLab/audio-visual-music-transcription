    def transcribeFrames(self,framesBatch, forcedStartPos= None, velocityCriteron = "hamming", onsetBound = None, candidateNotes=[]):
        device = framesBatch.device
        nBatch=  framesBatch.shape[0]
        crfBatch, ctxBatch = self.processFramesBatch(framesBatch)
        nSymbols = len(self.targetMIDIPitch)




        path = crfBatch.decode(forcedStartPos = forcedStartPos, forward=False)

        assert(nSymbols*nBatch == len(path))

        # also get the last position for each path for forced decoding
        # endPos = 
        if onsetBound is not None:
            path = [[e for e in _ if e[0]<onsetBound] for _ in path]
        
        # lastP = []

        # for curP in path:
        #     if len(curP) == 0:
        #         lastP.append(0)
        #     else:
        #         lastP.append(curP[-1][1])
        lastP = [curP[-1][1] if curP else 0 for curP in path]




        # then predict attributes associated with frames

        # obtain segment features

        nIntervalsAll =  sum([len(_) for _ in path])
        if nIntervalsAll == 0:
            # nothing detected, return empty
            return [[] for _ in range(nBatch)], lastP
        # print("#e:", nIntervalsAll)

        # intervalsBatch = []
        # for idx in range(nBatch):
        #     curIntervals =  path[idx*nSymbols: (idx+1)*nSymbols]
        #     intervalsBatch.append(curIntervals)
        intervalsBatch = [path[idx * nSymbols: (idx + 1) * nSymbols] for idx in range(nBatch)]
       

        # then predict the attribute set

        ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)

        pitchEmbed_all = self.pitchEmbedding(symIdx_all)

        attributeInput = torch.cat([ctx_a_all,
                                   ctx_b_all,
                                   ctx_a_all*ctx_b_all,
                                   pitchEmbed_all],dim = -1)



        logitsVelocity = self.velocityPredictor(attributeInput)
        pVelocity = F.softmax(logitsVelocity, dim = -1)

        
        #MSE
        if velocityCriteron == "mse":
            w = torch.arange(128, device = device)
            velocity = (pVelocity*w).sum(-1)
        elif velocityCriteron == "match":
            #TODO: Minimal risk 
            # predict velocity, readout by minimizing the risk
            # 0.1 is usually the tolerance for the velocity, so....
            
            # It will never make so extreme predictions

            # create the risk matrix
            w = torch.arange(128, device = device)

            # [Predicted, Actual]

            tolerance = 0.1* 128
            utility = ((w.unsqueeze(1)- w.unsqueeze(0)).abs()<tolerance).float()

            r = pVelocity@utility

            velocity = torch.argmax(r, dim = -1)


        elif velocityCriteron == "hamming":
            # return the mode
            velocity = torch.argmax(pVelocity, dim = -1)

        elif velocityCriteron == "mae":
            # then this time return the median
            pCum = pVelocity.cumsum(-1)
            tmp = (pCum-0.5)>0
            w2 = torch.arange(128, 0. ,-1, device = device)

            velocity = torch.argmax(tmp*w2, dim = -1)


        else:
            raise Exception("Unrecognized criterion: {}".format(velocityCriteron))

        



        ofValue = self.refinedOFPredictor(attributeInput)
        # ofValue = torch.sigmoid(ofValue)-0.5
        ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

        ofValue = (ofDist.mean-0.5)/0.99
        ofValue = torch.clamp(ofValue, -0.5, 0.5)

        # print(velocity)
        # print(ofValue)

        # generate the final result


        # parse the list of path to (begin, end, midipitch, velocity) 


        velocity = velocity.cpu().detach().tolist()
        ofValue = ofValue.cpu().detach().tolist()

        assert(len(velocity) == len(ofValue))
        assert(len(velocity) == nIntervalsAll)
         

        nCount = 0 

        notes = [[] for _ in range(nBatch)]

        frameDur = self.hopSize/self.fs


        # ADJUST TARGETMIDIPITCH
        # Check per second ##########################
        # for idx in range(len(listoflists)): or instead nBatch = len(listoflists)
        # for idx in range(len(candidateNotes)):
        for idx, curIntervals in enumerate(intervalsBatch):
            # curIntervals = intervalsBatch[idx]
            allowedPitches = set(candidateNotes[idx]) if idx < len(candidateNotes) else set()
            #print(range(nBatch))
            # Check the correct notes per second. Nope this is start and end times
            #curIntervals = candidateNotes[idx]

            # Find the right MIDI pitch based on the batch number ############## enumerate(candidateNotes[idx])

            # Dynamically change this line
            for j, eventType in enumerate(self.targetMIDIPitch):
                # Process if eventType is allowed or no restirctions
                if eventType in allowedPitches or not allowedPitches: 
                    lastEnd = 0
                    for aInterval in curIntervals[j]:
                        # print(aInterval, eventType, velocity[nCount], ofValue[nCount])
                        
                        curVelocity = velocity[nCount]

                        curOffset = ofValue[nCount]
                        start = (aInterval[0]+ curOffset[0] )*frameDur
                        end = (aInterval[1]+ curOffset[1])*frameDur

                        start = max(start, lastEnd)
                        end = max(end, start+1e-8)
                        lastEnd = end
                        curNote = Note(
                            start = start,
                            end = end,
                            pitch = eventType,
                            velocity = curVelocity)
                        
                        notes[idx].append(curNote)

                        nCount+= 1


            notes[idx].sort(key=lambda note: (note.start, note.end, note.pitch))

        # sort all 
        # print(notes)

        return notes, lastP



    # Change the segement size and hop size to match-up with the visual model
    def transcribe(self, x, stepInSecond = 1, segmentSizeInSecond = 4, discardSecondHalf=False, candidateNotes=[]):



        x= x.transpose(-1,-2)
        padTimeBegin = (segmentSizeInSecond-stepInSecond)

        x = F.pad(x, (math.ceil(padTimeBegin*self.fs), math.ceil(self.fs* (padTimeBegin))))

        nSample = x.shape[-1]
        

        eventsAll= []

        startFrameIdx = math.floor(padTimeBegin*self.fs/self.hopSize)
        startPos = [startFrameIdx]* len(self.targetMIDIPitch)
        # startPos =None

        stepSize = math.ceil(stepInSecond*self.fs/self.hopSize)*self.hopSize
        segmentSize = math.ceil(segmentSizeInSecond*self.fs)

        for i, step in enumerate(range(0, nSample, stepSize)):
            # t1 = time.time()

            # j = min(i+ segmentSize, nSample)
            j = min(step + segmentSize, nSample)
            # print(i, j)


            # beginTime = (i)/ self.fs -padTimeBegin
            beginTime = (step) / self.fs - padTimeBegin
            # print(beginTime)

            # curSlice = frames[:, i:j, :]
            # curSlice = x[:, i:j]
            curSlice = x[:, step:j]
            curFrames = makeFrame(curSlice, self.hopSize, self.windowSize)

            # # print(curSlice.shape)
            # # print(startPos)
            # startPos = None
            if discardSecondHalf:
                onsetBound = stepSize
            else:
                onsetBound = None

            # Select the right candidate notes (based on the segment)    
            segmentCandidateNotes = candidateNotes[i] if i < len(candidateNotes) else []

            curEvents, lastP = self.transcribeFrames(curFrames.unsqueeze(0), forcedStartPos = startPos, velocityCriteron = "hamming", onsetBound= onsetBound, candidateNotes=candidateNotes)
            curEvents = curEvents[0]


            startPos = []
            for k in lastP:
                startPos.append(max(k-int(stepSize/self.hopSize), 0))

            # # shift all notes by beginTime
            for e in curEvents:
                if isinstance(e, Note):
                    e.start += beginTime
                    e.end  += beginTime 

                    e.start = max(e.start, 0)
                    e.end = max(e.end, e.start+1e-5)




            # t2 = time.time()
            # # print("elapsed:", t2-t1)

            eventsAll.extend(curEvents)


        # check overlapping of eventsAll after adding the refined position

        eventsAll = resolveOverlapping(eventsAll)




        return eventsAll


