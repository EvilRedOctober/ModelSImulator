import sys
from datetime import datetime
from collections import deque

from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation

from MS2d_models import MODELS
from MS2d_form import Ui_MainWindow

DELAYS = (-1, 2000, 1000, 500, 250, 125, 50)  # Times for delay
SPEED = ('пауза', 'очень низкая', 'низкая', 'средняя',
         'высокая', 'очень высокая', 'максимальная')


class Model_Thread(QtCore.QThread):
    paintingSignal = QtCore.pyqtSignal(QtGui.QImage)
    graphSignal = QtCore.pyqtSignal(list)
    simEndedSignal = QtCore.pyqtSignal()

    def __init__(self, model, time):
        QtCore.QThread.__init__(self)
        self.model = model
        self.size = 0
        self.time = time
        self.running = False
        self.img = QtGui.QImage(1, 1,
                                QtGui.QImage.Format_ARGB32_Premultiplied)

    def new_start(self, model, size):
        super().start()
        self.model = model
        self.size = size

    def run(self):
        self.running = True
        self.paint()
        self.paintingSignal.emit(self.img.copy())
        while self.running and not self.model.is_ended():
            if self.time != -1:
                self.msleep(self.time)
                values = self.model.step()
                self.paint()
                self.paintingSignal.emit(self.img.copy())
                self.graphSignal.emit(values)
            else:
                self.msleep(1000)
        if self.model.is_ended():
            self.simEndedSignal.emit()

    def paint(self):
        self.img.fill(QtGui.QColor("#ffffff").rgb())
        painter = QtGui.QPainter()
        painter.begin(self.img)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor('#ffffff'), 0, style=QtCore.Qt.NoPen))
        for i in range(self.size):
            for j in range(self.size):
                color = self.model.get_color(i, j)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(color),
                                              style=QtCore.Qt.SolidPattern))
                painter.drawRect(QtCore.QRect(i*8 - 1, j*8 - 1, 7, 7))
        painter.end()

    def set_time(self, pos):
        self.time = DELAYS[pos]


class Paint_Widget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parameters for visualisation
        self.setFixedSize(QtCore.QSize(1, 1))
        self.img = QtGui.QImage(1, 1,
                                QtGui.QImage.Format_ARGB32_Premultiplied)

    @QtCore.pyqtSlot(QtGui.QImage)
    def change_image(self, img):
        self.img = img
        self.update()
        self.repaint()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.img)


class My_Canvas(FigureCanvas):
    """Widget to show graph"""
    def __init__(self, parent, x, Y, legends):
        self.fig = Figure(figsize=(5, 5))
        self.x = x
        self.Y = Y
        self.legends = legends
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.ax = self.figure.add_subplot()
        self.anim = animation.FuncAnimation(self.fig, self.anim_step, interval=300)
        self.draw()

    def anim_step(self, data=None):
        """Updating graph"""
        self.ax.clear()
        for i in range(len(self.Y)):
            self.ax.plot(self.x, self.Y[i], label=self.legends[i])
        if self.Y:
            self.ax.legend()
        self.ax.grid()
        self.ax.relim()  # update axes limits
        self.ax.autoscale_view(True, True, True)
        self.show()


class ParametersDialog(QtWidgets.QDialog):
    def __init__(self, parameters, proxy, parent=None):
        # Main part of sub window
        QtWidgets.QDialog.__init__(self, parent)

        self.setWindowTitle("Параметры модели")
        self.resize(500, 300)
        self.mainBox = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainBox)

        # Table with parameters
        self.tableWidget = QtWidgets.QTableWidget()
        self.mainBox.addWidget(self.tableWidget)
        n = len(parameters)
        self.tableWidget.setRowCount(n)
        self.tableWidget.setColumnCount(2)
        # Making headers fo table
        item = QtWidgets.QTableWidgetItem('Параметр')
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem('Значение')
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        for i, key in enumerate(parameters):
            # Setting widgets
            label = QtWidgets.QLabel(parameters[key]['name_rus'])
            spin = QtWidgets.__dict__[parameters[key]['spin_type']]()
            spin.setValue(parameters[key]['value'])
            spin.setMaximum(parameters[key]['max'])
            spin.setMinimum(parameters[key]['min'])

            # Creating functions for signals
            def callback(value, k=key):
                proxy[k] = value

            if parameters[key]['spin_type'] == 'QDoubleSpinBox':
                spin.setDecimals(3)
                spin.valueChanged[float].connect(callback)
                spin.setSingleStep((parameters[key]['max'] - parameters[key]['min'])/20)
            else:
                spin.valueChanged[int].connect(callback)
            self.tableWidget.setCellWidget(i, 0, label)
            self.tableWidget.setCellWidget(i, 1, spin)
        self.tableWidget.resizeColumnsToContents()

        # Button OK
        self.btnOK = QtWidgets.QPushButton("&OK")
        self.box = QtWidgets.QDialogButtonBox(QtCore.Qt.Horizontal)
        self.box.addButton(self.btnOK, QtWidgets.QDialogButtonBox.AcceptRole)
        self.box.accepted.connect(self.accept)
        self.mainBox.addWidget(self.box)

        self.update()


class my_window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        # Setup
        super(my_window, self).__init__()
        self.setupUi(self)

        # Widgets setups
        self.tabWidget.setCurrentIndex(0)
        self.setWindowIcon(QtGui.QIcon("./images/MD2d.jpg"))
        self.modelSelector.addItems(MODELS.keys())
        self.ModelClass = MODELS[self.modelSelector.currentText()]
        self.size = self.spinSize.value()
        if self.ModelClass:
            self.model = self.ModelClass(self.size)
        self.gridLayout.setSpacing(1)
        self.widget.setStyleSheet("""background-color: white;
                                     border-style: outset;
                                     border-width: 1px;
                                     border-color: black;
                                  """)
        self.thread = Model_Thread(self.model, 1000)
        w = Paint_Widget()
        self.gridLayout.addWidget(w, 0, 0)
        self.x = deque([], maxlen=30)
        self.Y = []
        self.legends = []
        self.canvas = My_Canvas(self.tab_graph, self.x, self.Y, self.legends)
        self.canvas.move(0, 0)
        self.gridLayout_3.addWidget(self.canvas, 0, 0)
        self.parameters = {}
        self.proxy_parameters_value = {}

        # Connecting function to events
        self.gridLayout.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.btnStart.clicked.connect(self.click_start)
        self.modelSelector.activated.connect(self.model_selected)
        self.horizontalSlider.valueChanged[int].connect(self.on_value_changed)
        self.horizontalSlider.valueChanged[int].connect(self.thread.set_time)
        self.thread.paintingSignal[QtGui.QImage].connect(w.change_image)
        self.thread.graphSignal[list].connect(self.graph_change)
        self.thread.simEndedSignal.connect(self.on_ended)

        # Setup menu bar
        self.actExit = QtWidgets.QAction("Выход", None)
        self.actExit.setShortcut("Ctrl+Q")
        self.actExit.triggered.connect(QtWidgets.qApp.quit)
        self.menuHelp = QtWidgets.QMenu("Информация")
        self.actParameters = QtWidgets.QAction("Параметры", None)
        self.actParameters.setShortcut("F3")
        self.actParameters.triggered.connect(self.click_parameters)
        self.actAbout = QtWidgets.QAction("О программе", None)
        self.actAbout.setShortcut("F1")
        self.actAbout.triggered.connect(self.click_about)
        self.actPyQt = QtWidgets.QAction("О Qt", None)
        self.actPyQt.setShortcut("F2")
        self.actPyQt.triggered.connect(self.click_pyqt)
        self.menuHelp.addAction(self.actAbout)
        self.menuHelp.addAction(self.actPyQt)
        self.menuBar.addAction(self.actParameters)
        self.menuBar.addMenu(self.menuHelp)
        self.menuBar.addAction(self.actExit)

        # This function runs at beginning of program
        self.model_selected()
        self.on_value_changed(3)

    def click_start(self):
        """When starting modelling"""
        self.thread.running = False
        # Wait till thread stopped
        while self.thread.isRunning():
            pass
        # Changing size of grid
        self.size = self.spinSize.value() * 2 + 1
        self.gridLayout.itemAt(0).widget().setFixedSize(8*self.size-1, 8*self.size-1)
        self.thread.img = QtGui.QImage(self.size * 8 - 1, self.size * 8 - 1,
                                       QtGui.QImage.Format_ARGB32_Premultiplied)
        # and updating it
        self.gridLayout.update()
        self.widget.update()
        self.widget.repaint()
        QtCore.QCoreApplication.processEvents()
        # Reset data for graph
        self.x.clear()
        self.Y.clear()
        self.legends.clear()
        self.x.append(1)
        # Preparing model and changing status
        self.model = self.ModelClass(self.size, **self.proxy_parameters_value)
        self.statusBar().showMessage(self.modelSelector.currentText())
        # Get data about legend and values
        graph_data = self.model.start()
        for value, legend in graph_data:
            self.legends.append(legend)
            self.Y.append(deque([value], maxlen=30))
        self.thread.new_start(self.model, self.size)
        # Print to history
        time = "<b>" + datetime.now().strftime("%d.%m.%Y %H:%M:%S") + "</b>"
        size = ". Размер поля {0} на {0}.".format(self.size)
        self.textHistory.append(time + " Начата симуляция \"" + self.modelSelector.currentText() + "\"" + size)

    def model_selected(self):
        """When user select model from ComboBox"""
        self.ModelClass = MODELS[self.modelSelector.currentText()]
        if self.ModelClass:
            self.modelText.setText(self.ModelClass.MODEL_TEXT)
            self.parameters = self.ModelClass.PARAMETERS
            self.proxy_parameters_value = {key: self.parameters[key]['value'] for key in self.parameters}
        else:
            self.modelText.setText('Модель еще не готова!')

    def on_value_changed(self, pos):
        """If changing speed of simulation"""
        self.label.setText('Скорость симуляции: ' + SPEED[pos])

    @QtCore.pyqtSlot()
    def on_ended(self):
        time = "<b>" + datetime.now().strftime("%d.%m.%Y %H:%M:%S") + "</b>"
        self.textHistory.append(time + " Симуляция закончена.")

    @QtCore.pyqtSlot(list)
    def graph_change(self, values):
        self.x.append(self.x[-1] + 1)  # update data
        for i, value in enumerate(values):
            self.Y[i].append(value)

    def click_about(self):
        QtWidgets.QMessageBox.information(self, "О программе",
                                          "Программа 'Симулятор двумерных моделей', Бакаев А.И., 2021\n\n"
                                          "Основное назначение - запуск симуляций разработанных двумерных моделей. "
                                          "Программа содержит множество моделей, достпуных для симуляции. Доступен "
                                          "выбор размера модели и скорости симуляции. Имеется возможность менять "
                                          "параметры выбранной модели с помощью соответствующего пункта меню. \n\n"
                                          "Программа разработана на языке Python при использовании библиотеки PyQt.",
                                          buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

    def click_pyqt(self):
        QtWidgets.QMessageBox.aboutQt(self)

    def click_parameters(self):
        for key in self.proxy_parameters_value:
            self.parameters[key]['value'] = self.proxy_parameters_value[key]
        dialog = ParametersDialog(self.parameters, self.proxy_parameters_value, self)
        dialog.exec()


app = QtWidgets.QApplication(sys.argv)
application = my_window()
application.show()

sys.exit(app.exec_())
