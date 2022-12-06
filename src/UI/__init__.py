from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtWidgets
from typing import Callable, Union
try:
    from UI.PYFile.mainPage import Ui_Form
    from UI.PYFile.tabLayout import tab
except:
    print('fail import ui')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from model.Hopfield import HopfieldNetwork
matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class mainPageWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, readData: Callable[[str], tuple[list[int], int]]):
        super(QtWidgets.QWidget, self).__init__()
        self.setupUi(self)
        self.resChart: FigureCanvasQTAgg = MplCanvas(self.sceneLayout)
        self.currentAns:Union[None, np.ndarray]=None
        self.sceneLayout.addWidget(self.resChart)
        self.resChart.axes.axis('off')
        self.model:HopfieldNetwork=HopfieldNetwork()
        self.readData:Callable[[str], tuple[list[int], int]]=readData
        self.TrainBtn.clicked.connect(self.trainBtnClicked)
        self.addNoiseBtn.clicked.connect(self.addNoiseBtnClicked)
        self.runUntilConvergeBtn.clicked.connect(self.runConvergeClicked)
        self.runStepsBtn.clicked.connect(self.runStepsClicked)
        self.converge:bool=True
        self.runStepsBtn.setEnabled(False)
        self.runUntilConvergeBtn.setEnabled(False)
        self.addNoiseBtn.setEnabled(False)

    def drawChart(self, chart:MplCanvas, data:np.ndarray[int], rowSize:int)->None:
        chart.axes.clear()
        chart.axes.axis('off')
        chart.axes.plot([0, rowSize], [0, 0], 'g')
        chart.axes.plot([0, rowSize], [data.shape[0]//rowSize, data.shape[0]//rowSize], 'g')
        chart.axes.plot([0, 0], [0, data.shape[0]//rowSize], 'g')
        chart.axes.plot([rowSize, rowSize], [0, data.shape[0]//rowSize], 'g')
        for ycoor in range(data.shape[0]//rowSize - 1, -1, -1):
            for xcoor in range(rowSize):
                if data[ycoor*rowSize+xcoor]==1:
                    chart.axes.add_patch(Rectangle((xcoor, data.shape[0]//rowSize-ycoor-1), 1, 1, facecolor = 'black', lw=0))
        chart.draw()

    def trainBtnClicked(self)->None:
        def setupChart(tab:tab, data:np.ndarray, size:int):
            tab.chart: FigureCanvasQTAgg = MplCanvas(self.sceneLayout)
            tab.chart.axes.axis('off')
            tab.verticalLayout.addWidget(tab.chart)
            tab.data=data
            tab.rowSize=size
            self.drawChart(tab.chart, tab.data, tab.rowSize)

        if self.dataset1RB.isChecked():
            dataset, size=self.readData('dataset/Basic_Training.txt')
            testset, _=self.readData('dataset/Basic_Testing.txt')
        elif self.dataset2RB.isChecked():
            dataset, size=self.readData('dataset/Bonus_Training.txt')
            testset, _=self.readData('dataset/Bonus_Testing.txt')
        else: return
        dataset, testset = np.array(dataset), np.array(testset)
        while self.tabWidget.count()!=0: 
            self.tabWidget.removeTab(0)
        for i in range(testset.shape[0]):
            self.tabWidget.addTab(tab(), f"data{i+1}")
            setupChart(self.tabWidget.widget(self.tabWidget.count()-1), testset[i], size)
        self.model.fit(dataset)
        self.runStepsBtn.setEnabled(True)
        self.runUntilConvergeBtn.setEnabled(True)
        self.addNoiseBtn.setEnabled(True)
        self.converge=True

    def addNoiseBtnClicked(self):
        currTab:tab=self.tabWidget.currentWidget()
        oldData:np.ndarray=currTab.data
        noise=np.random.choice(oldData.shape[0], 3, replace=False)
        oldData[noise]*=-1
        self.drawChart(currTab.chart, oldData, currTab.rowSize)
        currTab.data=oldData
        
    def runConvergeClicked(self):
        currTab:tab=self.tabWidget.currentWidget()
        ans=self.model.predict(currTab.data)
        self.drawChart(self.resChart, ans, currTab.rowSize)

    def runStepsClicked(self):
        currTab:tab=self.tabWidget.currentWidget()
        if self.currentAns is None or self.converge:
            self.currentAns=currTab.data.copy()
            self.converge=False
        for _ in range(self.stepCountSB.value()):
            self.currentAns, hater = self.model.next(self.currentAns)
        if hater==-1: self.converge=True
        self.drawChart(self.resChart, self.currentAns, currTab.rowSize)
        
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, readData: Callable[[str], tuple[list[int], int]]):
        super(MainWindow, self).__init__()
        mp: QtWidgets.QWidget = mainPageWidget(readData)
        self.setCentralWidget(mp)
        self.show()


class App():
    def __init__(self, readData: Callable[[str], tuple[list[int], int]]):
        self.readData: Callable[[str], list[str]] = readData

    def startAppSync(self):
        import sys
        app = QtWidgets.QApplication(sys.argv)
        w = MainWindow(self.readData)
        app.exec()