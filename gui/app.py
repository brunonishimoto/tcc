# Simple enough, just import everything from tkinter.
from tkinter import *
from utils import *
import tailer as t
import re
import time
import json

#download and install pillow:
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow
from PIL import Image, ImageTk

class Application:
    def __init__(self, master=None):
        self.master = master

        self.frame_left = Frame(self.master, bg="#aaaaaa", borderwidth=1, relief=SOLID)
        # self.frame_left.grid(column=0, row=0)
        self.frame_left.pack(side=LEFT, fill=Y)
        self.frame_right = Frame(self.master, bg="#aaaaaa", borderwidth=1, relief=SOLID)
        # self.frame_right.grid(column=1, row=0)
        self.frame_right.pack(side=RIGHT, fill=Y)

        self.user = self.load_user()
        self.dm = self.load_dm()
        self.nlu = self.load_nlu()
        self.nlg = self.load_nlg()
        self.image = self.load_image()
        self.goal = self.load_goal()
        self.db = self.load_db()
        self.read_log()

    def read_log(self):
        logs = t.tail(open("../logs/dialogue_logger.log", "r"), 8)

        dict_infos = {}

        for log in (logs):
            log = re.sub(r'INFO: (\d{2}/){2}\d{4} (\d{2}:?){3} (AM|PM) ', '', log)
            log = log.replace("'", '"')
            if log:
                info, value = log.split(':', 1)
                dict_infos[info] = value

        turn = int(json.loads(dict_infos.get('Agent action', "{}")).get('round', 0))

        self.image["text"] = f"Turno {turn}"
        self.user["text"] = dict_infos.get('User sentence', "")
        self.nlu["text"] = action_to_string(dict_infos.get('User action', "{}"))
        self.dm["text"] = action_to_string(dict_infos.get('Agent action', "{}"))
        self.nlg["text"] = dict_infos.get('Agent sentence', "")

        if not self.goal["text"] or self.goal["text"] == '{}':
            self.goal["text"] = goal_to_string(dict_infos.get('Goal', "{}"))

        self.db["text"] = informs_to_string(dict_infos.get('Current informs', "{}"), dict_infos.get('DB count', "{}"))

        self.master.after(1000, self.read_log)

    def load_user(self):
        user = Label(self.frame_left,
                #    bg = "red",
                   text="",
                   font="Helvetica 16",
                   wraplength=300)

        # user.pack(side=LEFT)
        user.grid(column=0, row=1)

        return user

    def load_nlu(self):
        nlu = Label(self.frame_left,
                #    bg = "red",
                   text="",
                   font="Helvetica 16",
                   wraplength=830)

        # nlu.pack(side=TOP)
        nlu.grid(row=0, column=1)

        return nlu

    def load_dm(self):
        dm = Label(self.frame_left,
                #    bg = "red",
                   text="",
                   font="Helvetica 16",
                   wraplength=300)

        # dm.pack(side=RIGHT)
        dm.grid(row=1, column=2)

        return dm

    def load_nlg(self):
        nlg = Label(self.frame_left,
                #    bg = "red",
                   text="",
                   font="Helvetica 16",
                   wraplength=830)

        # nlg.pack(side=BOTTOM)
        nlg.grid(row=2, column=1)

        return nlg

    def load_image(self):
        load = Image.open("chatbot.png")
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        img = Label(self.frame_left,
                    image=render,
                    compound=CENTER,
                    text="",
                    # bg = "green",
                    font="Helvetica 28 bold")
        img.image = render
        # img.pack()
        img.grid(row=1, column=1)

        return img

    def load_goal(self):
        goal = Label(self.frame_right,
                     text="",
                    #  bg="yellow",
                     borderwidth=3,
                     relief=RIDGE,
                     height=10,
                     anchor="n",
                     font="Helvetica 16",
                     wraplength=600)
        # goal.grid(row=0)
        goal.pack(side=TOP, fill=X)

        return goal

    def load_db(self):
        db = Label(self.frame_right,
                   text="",
                #    bg="blue",
                   borderwidth=3,
                   relief=RIDGE,
                   anchor="s",
                   font="Helvetica 16",
                   wraplength=600)

        button = Button(self.frame_right,
                        text="Ver resultados",
                        font="Helvetica 16",
                        command=self.load_db_results)

        # db.grid(row=1, rowspan=3)
        button.pack(side=BOTTOM)
        db.pack(side=BOTTOM, fill=X)

        return db

    def load_db_results(self):
        new_window = Toplevel(self.master, width=700)
        new_window.title('Ingressos disponíveis')

        scrollbar = Scrollbar(new_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        tickets_list = Text(new_window, yscrollcommand=scrollbar.set, font="Helvetica 16")

        db_results_list = self.get_db_results()

        if db_results_list:
            tickets_list.insert(END, f"Ingressos disponíveis\n\n")
        else:
            tickets_list.insert(END, f"Não há ingressos disponíveis\n\n")

        for idx, result in enumerate(db_results_list):
            tickets_list.insert(END, f"{idx} - {result}\n\n")

        tickets_list.configure(state=DISABLED)

        tickets_list.pack(side=LEFT, fill=BOTH)
        scrollbar.config(command=tickets_list.yview)


    def get_db_results(self):
        logs = t.tail(open("../logs/dialogue_logger.log", "r"), 8)

        dict_infos = {}
        for log in (logs):
            log = re.sub(r'INFO: (\d{2}/){2}\d{4} (\d{2}:?){3} (AM|PM) ', '', log)
            log = log.replace(" '", ' "')
            log = log.replace("':", '":')
            log = log.replace("',", '",')
            log = log.replace("'}", '"}')
            log = log.replace("{'", '{"')
            info, value = log.split(':', 1)
            dict_infos[info] = value

        db_results = db_to_string(dict_infos['DB results'])
        # print(db_results)
        return db_results


# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = Tk()

# Changing the title of the root
root.title("Sistema de Diálogo")
root.geometry("1260x900")

app = Application(root)
root.mainloop()
