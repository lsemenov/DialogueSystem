# greeting_uterr - приветствие
# answer(inform) - ответ на общий вопрос
# response(request) - rag ответ на основе базы знаний
# confirm(denny) - уточняющий вопрос
# bye_end - заканчивает разговор
# forward - переадресация вопроса на 1 линию тех. поддержки
# reject - заканчивает разговор не знает ответа
# api_tools - действие со стороним API


class Action(object):

    def __init__(self):
        pass


class ActionUtterance(Action):

    def answer(self):
        pass

    def response(self):
        pass

    def ask_question(self):
        pass


    def api_tools(self):
        pass


    def forward(self):
        pass


class ActionApi(Action):
    def api_tools(self):
        pass



class ActionReject():
    def forward(self):
        pass


    def reject(self):
        pass



