# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MS2d_form.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.WindowModal)
        MainWindow.resize(900, 892)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(900, 821))
        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Book")
        font.setPointSize(12)
        MainWindow.setFont(font)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setAnimated(True)
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setDockNestingEnabled(True)
        MainWindow.setDockOptions(QtWidgets.QMainWindow.AllowNestedDocks|QtWidgets.QMainWindow.AllowTabbedDocks|QtWidgets.QMainWindow.AnimatedDocks)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(425, 0))
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.groupBox1 = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox1.sizePolicy().hasHeightForWidth())
        self.groupBox1.setSizePolicy(sizePolicy)
        self.groupBox1.setMinimumSize(QtCore.QSize(272, 509))
        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Medium")
        font.setPointSize(12)
        font.setItalic(False)
        self.groupBox1.setFont(font)
        self.groupBox1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.groupBox1.setFlat(False)
        self.groupBox1.setCheckable(False)
        self.groupBox1.setObjectName("groupBox1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label1 = QtWidgets.QLabel(self.groupBox1)
        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Book")
        font.setPointSize(12)
        font.setItalic(False)
        self.label1.setFont(font)
        self.label1.setObjectName("label1")
        self.verticalLayout.addWidget(self.label1)
        self.modelSelector = QtWidgets.QComboBox(self.groupBox1)
        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Book")
        font.setPointSize(12)
        font.setItalic(True)
        self.modelSelector.setFont(font)
        self.modelSelector.setObjectName("modelSelector")
        self.verticalLayout.addWidget(self.modelSelector)
        self.label2 = QtWidgets.QLabel(self.groupBox1)
        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Book")
        font.setPointSize(12)
        font.setItalic(False)
        self.label2.setFont(font)
        self.label2.setObjectName("label2")
        self.verticalLayout.addWidget(self.label2)
        self.modelText = QtWidgets.QTextBrowser(self.groupBox1)
        self.modelText.setObjectName("modelText")
        self.verticalLayout.addWidget(self.modelText)
        self.label3 = QtWidgets.QLabel(self.groupBox1)
        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Book")
        font.setPointSize(12)
        font.setItalic(False)
        self.label3.setFont(font)
        self.label3.setObjectName("label3")
        self.verticalLayout.addWidget(self.label3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.spinSize = QtWidgets.QSpinBox(self.groupBox1)
        self.spinSize.setMinimum(5)
        self.spinSize.setMaximum(30)
        self.spinSize.setObjectName("spinSize")
        self.horizontalLayout.addWidget(self.spinSize)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.btnStart = QtWidgets.QPushButton(self.groupBox1)
        self.btnStart.setObjectName("btnStart")
        self.verticalLayout.addWidget(self.btnStart)
        self.horizontalLayout_3.addWidget(self.groupBox1)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_cells = QtWidgets.QWidget()
        self.tab_cells.setObjectName("tab_cells")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_cells)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.widget = QtWidgets.QWidget(self.tab_cells)
        self.widget.setAutoFillBackground(True)
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2.addWidget(self.widget, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_cells, "")
        self.tab_graph = QtWidgets.QWidget()
        self.tab_graph.setObjectName("tab_graph")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_graph)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.tabWidget.addTab(self.tab_graph, "")
        self.horizontalLayout_3.addWidget(self.tabWidget)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(6)
        self.horizontalSlider.setPageStep(1)
        self.horizontalSlider.setProperty("value", 3)
        self.horizontalSlider.setTracking(True)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)
        self.horizontalSlider.setInvertedControls(False)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.verticalLayout_3.addWidget(self.horizontalSlider)
        self.groupBox2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox2.setMinimumSize(QtCore.QSize(400, 200))
        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Medium")
        font.setPointSize(12)
        font.setItalic(False)
        self.groupBox2.setFont(font)
        self.groupBox2.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.groupBox2.setFlat(False)
        self.groupBox2.setCheckable(False)
        self.groupBox2.setObjectName("groupBox2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.textHistory = QtWidgets.QTextBrowser(self.groupBox2)
        self.textHistory.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.textHistory.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textHistory.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.textHistory.setObjectName("textHistory")
        self.verticalLayout_2.addWidget(self.textHistory)
        self.verticalLayout_3.addWidget(self.groupBox2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 900, 27))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "?????????????????? ?????????????????? ??????????????"))
        self.groupBox1.setTitle(_translate("MainWindow", "?????????????????? ??????????????????????????"))
        self.label1.setText(_translate("MainWindow", "?????????? ????????????"))
        self.label2.setText(_translate("MainWindow", "?????????????? ????????????????"))
        self.label3.setText(_translate("MainWindow", "???????????? ????????"))
        self.btnStart.setText(_translate("MainWindow", "???????????? ??????????????????????????"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_cells), _translate("MainWindow", "?????????????????? ????????????"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_graph), _translate("MainWindow", "???????????? ??????????????????"))
        self.label.setText(_translate("MainWindow", "???????????????? ??????????????????"))
        self.groupBox2.setTitle(_translate("MainWindow", "??????????????"))
