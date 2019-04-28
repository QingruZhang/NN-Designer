######################################
# Author: 张庆儒
# SID: 516021910419
######################################
# Main file

from TopDesign import *


if __name__ == '__main__':
	app = QApplication(sys.argv)
	TopWindow = TopDesign()
	TopWindow.show()
	sys.exit(app.exec_())
	