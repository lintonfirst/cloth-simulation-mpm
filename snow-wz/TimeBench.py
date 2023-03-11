import time


class TimeBench:
    def __init__(self):
        self.phaseNameList = []
        self.timePointList = []

    def addPhase(self, phaseName):
        timePoint = time.time()
        self.phaseNameList.append(phaseName)
        self.timePointList.append(timePoint)

    def report(self, frame=None):
        self.addPhase("end")
        if frame is not None:
            print("\033[1;36mframe {}\033[0m".format(frame))
        for i in range(self.phaseNameList.__len__()-1):
            print("\033[1;36m\t{}: {}s\033[0m".format(self.phaseNameList[i], self.timePointList[i + 1] - self.timePointList[i]))
        print("\n")
        self.phaseNameList.clear()
        self.timePointList.clear()


timeBench = TimeBench()
