import pywinauto
from pywinauto import Application

# 일단 선행되어야 할건 print_control_identifiers()로
# 프로그램의 정보들을 다 뽑아내어야 하고
# 그거를 토대로 어떤 항목을 제어할지 결정하여야함.

# 실행시키고픈 프로그램의 절대경로를 적어준다.
# Application(backend='uia').start('프로그램 절대경로',
#                                  wait_for_idle=False)

# title_re에 적힌 title 이름을 가진 프로그램을 제어하기 위해 app에 담아줌.
# app = pywinauto.Desktop(backend="uia").window(title_re="타이틀 이름")

# 그 후 print_control_identifiers() 함수를 이용하여
# 앱이 어떤 것들로 이루어져있는지 파악해야함.
# print(app.print_control_identifiers())

# 앱의 버튼이름이 CONNECT이면 이거에 접근할려면
# app.CONNECT.click()로 접근하면 그 버튼을 한번 클릭함.

# app의 GUI 중 콤보박스가 있다면
# 콤보박스의 항목에 접근할려면
# app.ComboBox.select("항목이름")
# 이런식으로 접근해야함.

# app에 Edit창에 text를 입력하려면
# app.Edit.set_text("입력")

