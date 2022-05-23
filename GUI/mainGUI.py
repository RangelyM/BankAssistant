import sys
from collections import namedtuple
from typing import Tuple, List
from PyQt5.QtWidgets import  QTableWidgetItem
from PyQt5 import QtWidgets

from mydesign import Ui_MainWindow
from mydesign1 import Ui_MainWindow1
from mydesign3 import Ui_MainWindow3
from classification import *
import createCredit



class MyWindow(QtWidgets.QMainWindow):
    __hashData = {}
    COLUMNS = ['LIMIT_BAL', 'SEX', 'EDUCATION',
               'MARRIAGE', 'AGE', 'PAY_AMT1', 'PAY_AMT2',
               'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    LIMIT_BAL_CONST = 0.95

    def __init__(self):
        super(MyWindow, self).__init__()
        self.startMainWindow()

    def startMainWindow(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.btnClicked)

    def btnClicked(self):
        # self.window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow1()
        self.ui.setupUi(self)

        self.ui.buttonBox.accepted.connect(self.saveData)
        self.ui.buttonBox.rejected.connect(self.dropData)
        self.ui.pushButton_5.clicked.connect(self.nextTip)
        self.ui.pushButton_6.clicked.connect(self.back)
        self.ui.lineEdit.adjustSize()

    def saveData(self):

        self.__hashData[self.COLUMNS[0]] =int( self.ui.lineEdit.text())
        self.__hashData[self.COLUMNS[1]] = int(self.ui.comboBox.currentIndex())
        self.__hashData[self.COLUMNS[2]] = int(self.ui.comboBox_2.currentIndex())
        self.__hashData[self.COLUMNS[3]] = int(self.ui.comboBox_3.currentIndex())
        self.__hashData[self.COLUMNS[4]] = int(self.ui.lineEdit_2.text())
        self.__hashData[self.COLUMNS[5]] = int( self.ui.textEdit_6.text())
        self.__hashData[self.COLUMNS[6]] = int(self.ui.textEdit_7.text())
        self.__hashData[self.COLUMNS[7]] = int(self.ui.textEdit_8.text())
        self.__hashData[self.COLUMNS[8]] = int(self.ui.textEdit_9.text())
        self.__hashData[self.COLUMNS[9]] = int(self.ui.textEdit_10.text())
        self.__hashData[self.COLUMNS[10]] = int(self.ui.textEdit_13.text())

        self.nextTipLogRegr()

    def dropData(self):
        self.__hashData.clear()
        self.ui.lineEdit.clear()
        self.ui.lineEdit_2.clear()
        self.ui.textEdit_6.clear()
        self.ui.textEdit_7.clear()
        self.ui.textEdit_8.clear()
        self.ui.textEdit_9.clear()
        self.ui.textEdit_10.clear()
        self.ui.textEdit_13.clear()

    def back(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.btnClicked)

    def nextTip(self):

        self.ui = Ui_MainWindow3()
        self.ui.setupUi(self)

        answer = "Reject"
        if self.result:
            answer = "Accept"

        self.ui.label.setText("Neural Network procecc : ".join( answer ))


        self.ui.pushButton.clicked.connect(self.back1)
        self.ui.pushButton_2.clicked.connect(self.calc)
        self.ui.tableWidget.setItem(0)


    def calc(self):
        sum = int(self.ui.textEdit_5.text())
        procent = int(self.ui.textEdit_6.text())
        years = int(self.ui.textEdit_4.text())
        income = int(self.ui.textEdit_7.text())

        self.CreditPlan = createCredit.CreateCredit(sum,self.COLUMNS[4], 0.75 * income)

        self.creditPlan = self.CreditPlan.calc_results()

        self.ui.tableWidget(0, 0,QTableWidgetItem(self.creditPlan['Month credit'][0]))
        self.ui.tableWidget(1,0 ,QTableWidgetItem(self.creditPlan['Bank percent'][0]))
        self.ui.tableWidget(2,0, QTableWidgetItem(self.creditPlan['Sum of credit'][0]))
        self.ui.tableWidget(3, 0, QTableWidgetItem(self.creditPlan['Payment per month'][0]))

    def prepare_data(self, data: dict) -> np.ndarray:
        df = pd.DataFrame(columns=self.COLUMNS)
        for key, val in data.items():
            df[key] = [val]
        return np.array(df)[0]

    def nextTipLogRegr(self):
        data = self.prepare_data(self.__hashData)

        model = Classification(11, 1)
        model.load_state_dict(torch.load('bin/log_regr-UCI_13_rub.pt'))
        model.eval()

        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = torch.tensor(data, requires_grad=True, dtype=torch.float).cuda()
            else:
                inputs = torch.tensor(data, requires_grad=True, dtype=torch.float)
            output = model.forward(inputs)
        self.result = bool(round(output.data.item()))

    def nextTipNonLin(self):
        # data = self.prepare_data(self.__hashData)
        #
        # model = Classification(11, 1)
        # model.load_state_dict(torch.load('bin/non_lin-UCI_13_rub.pt'))
        # model.eval()
        #
        # with torch.no_grad():
        #     if torch.cuda.is_available():
        #         inputs = torch.tensor(data, requires_grad=True, dtype=torch.float).cuda()
        #     else:
        #         inputs = torch.tensor(data, requires_grad=True, dtype=torch.float)
        #     output = model.forward(inputs)
        # result = bool(round(output.data.item()))

        self.ui = Ui_MainWindow3()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.back1)

    def getRecommendations(self, pars: torch.nn.parameter.Parameter, data: np.ndarray) -> Tuple[List[str], int]:
        Par = namedtuple('Par', ['val', 'name'])
        par_col = []
        for par, col in zip(list(pars)[-2][0].tolist(), self.COLUMNS):
            par_col.append(Par(par, col))
        par_col.sort(reverse=True)

        # ТОП-3 параметра, которые не позволяют выдать кредит:
        bad_pars = [par.name for par in par_col if par.name != 'SEX'][:3]

        model = Classification(11, 1)
        model.load_state_dict('bin/log_regr-UCI_13_rub.pt')
        model.eval()

        result = False
        while not result:
            data[0] *= self.LIMIT_BAL_CONST
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs = torch.tensor(data, requires_grad=True, dtype=torch.float).cuda()
                else:
                    inputs = torch.tensor(data, requires_grad=True, dtype=torch.float)
                output = model.forward(inputs)
            result = bool(round(output.data.item()))

        # Сумма, на которую можно выдать кредит с текущими параметрами
        good_sum = round(data[0])

        return bad_pars, good_sum

    def back1(self):

        self.ui = Ui_MainWindow1()
        self.ui.setupUi(self)
        self.ui.pushButton_5.clicked.connect(self.nextTipNonLin)
        self.ui.pushButton_6.clicked.connect(self.back)
        self.ui.lineEdit.adjustSize()


app = QtWidgets.QApplication([])
application = MyWindow()
application.show()

sys.exit(app.exec())
