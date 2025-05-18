import customtkinter as ctk
from tkinter import filedialog,messagebox,ttk
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from pandasgui import show
from sklearn.preprocessing import StandardScaler, LabelEncoder
from io import StringIO
import re
import seaborn as sns
class DataPreprocessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Data Preprocessor")
        self.geometry("900x750")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("dark-blue")
        self.create_widgets()
    global data
    data = pd.DataFrame([])
    def create_widgets(self):
        self.filename = None
        #--------------Label & Text Area------------------
        self.label = ctk.CTkLabel(self,text="Processes On Data",font= ("Arial",20))
        self.label.place(rely=0.02, relx=0.15)
        self.text_area = ctk.CTkTextbox(self,height=150,width=400)
        self.text_area.place(rely= 0.08 , relx= 0.03)
        self.label_Machine = ctk.CTkLabel(self,text="Machine Learning (Soon)",font= ("Arial",20))
        self.label_Machine.place(rely=0.02, relx=0.63)
        self.text_area_Machine = ctk.CTkTextbox(self,height=150,width=400)
        self.text_area_Machine.place(rely= 0.08 , relx= 0.53)
        self.text_area_Machine.configure(state=tk.DISABLED)
        self.text_area.configure(state=tk.DISABLED)
        #-------------------------------------------------

        #-----------------Buttons-------------------------
        button_to_select = ctk.CTkButton(self, text = "Browse", fg_color = "blue", command = self.browse)
        button_to_select.place(rely=0.3,relx=0.17)
        button_to_load = ctk.CTkButton(self, text = "Load", fg_color = "green", command = self.load_data,width=50)
        button_to_load.place(rely=0.3,relx=0.4)
        self.process_button = ctk.CTkButton(self, text="Process Data",fg_color="red", command=self.process_data)
        self.process_button.place(rely=0.95,relx=0.17)
        self.save_button = ctk.CTkButton(self, text="Save",fg_color="green", command=self.save_file,width=50)
        self.save_button.place(rely=0.95,relx=0.01)
        self.process_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        #-------------------------------------------------

        #-----------------Dict of Options-----------------
        self.preprocess_options = {
                "Descripe Numerical Columns": False,
                "Descripe Categorical Columns": False,
                "Show 10 tail rows" : False,
                "Show 10 head rows" : False,
                "Scaling" : False,
                "Show Random Rows" : False,
                "Deal with Nulls" : False,
                "Method":"",
                "Number Or String" : False,
                "Encoding" : False,
                "Encoding_type":"",
                "Plot":False,
                "Choise_1":"",
                "Choise_2":"",
                "Choise_3":"",
                "info":False,
                "Drop":False,
            }
        #--------------------------------------------------

        #------------------Random Rows---------------------
        self.textbox = ctk.CTkEntry(self, placeholder_text="Enter Number of Samples",width=160)
        self.textbox.place(rely=0.75,relx=0.01)
        self.checkbox_var_2 = tk.BooleanVar()
        self.option1 = ctk.CTkCheckBox(self, text="Show Random Rows",variable=self.checkbox_var_2, command=self.Random_Rows)
        self.option1.place(rely=0.7,relx=0.01)
        self.textbox.configure(state=tk.DISABLED)
        self.option1.configure(state=tk.DISABLED)
        #--------------------------------------------------

        #---------------------Info-------------------------
        self.option2 = ctk.CTkCheckBox(self, text="Info About DataFrame", command=self.update_options)
        self.option2.place(rely=0.65,relx=0.01)
        self.option2.configure(state=tk.DISABLED)
        #--------------------------------------------------


        #----------------------Drop-----------------------
        self.option_drop = ctk.CTkCheckBox(self, text="Drop Columns", command=self.Drop_Func)
        self.option_drop.place(rely=0.35,relx=0.01)
        self.menu_button_3 = ctk.CTkButton(self, text="Select Columns", command=lambda stt = "Drop" : self.show_menu(stt))
        self.menu_button_3.place(rely=0.35,relx=0.15)
        self.selected_options_Drop = set()
        self.menu_button_3.configure(state=tk.DISABLED)
        self.option_drop.configure(state=tk.DISABLED)
        #--------------------------------------------------

        #----------------------Describe--------------------
        self.option3 = ctk.CTkCheckBox(self, text="Describe Numerical Columns", command=self.update_options)
        self.option3.place(rely=0.4,relx=0.01)
        self.option4 = ctk.CTkCheckBox(self, text="Describe Categorical Columns", command=self.update_options)
        self.option4.place(rely=0.45,relx=0.01)
        self.option3.configure(state=tk.DISABLED)
        self.option4.configure(state=tk.DISABLED)
        #--------------------------------------------------

        #----------------------Head&Tail-------------------
        self.option5 = ctk.CTkCheckBox(self, text="Show 10 tail rows", command=self.update_options)
        self.option5.place(rely=0.5,relx=0.01)
        self.option6 = ctk.CTkCheckBox(self, text="Show 10 head rows", command=self.update_options)
        self.option6.place(rely=0.55,relx=0.01)
        self.option5.configure(state=tk.DISABLED)
        self.option6.configure(state=tk.DISABLED)
        #--------------------------------------------------

        #---------------------Scaling----------------------
        self.option7 = ctk.CTkCheckBox(self, text="Scaling", command=self.update_options)
        self.option7.place(rely=0.6,relx=0.01)
        self.option7.configure(state=tk.DISABLED)
        #----------------------Describe--------------------

        #---------------------Nulls------------------------
        self.selected_options_Null = set()
        self.checkbox_var = tk.BooleanVar()
        self.option10 = ctk.CTkCheckBox(self, text="Deal with Nulls", variable=self.checkbox_var, command=self.toggle_radio_buttons)
        self.option10.place(rely=0.4,relx=0.33)
        self.radio_var = tk.StringVar()
        self.option10.configure(state=tk.DISABLED)
        self.radio_button1 = ctk.CTkRadioButton(self, text="Remove missing values", variable=self.radio_var, value="remove",command=self.toggle_radio_buttons_2)
        self.radio_button1.place(rely=0.45,relx=0.33)
        self.radio_button2 = ctk.CTkRadioButton(self, text="Fill with median", variable=self.radio_var, value="median",command=self.toggle_radio_buttons_2)
        self.radio_button2.place(rely=0.5,relx=0.33)
        self.radio_button3 = ctk.CTkRadioButton(self, text="Fill with previous value", variable=self.radio_var, value="ffill",command=self.toggle_radio_buttons_2)
        self.radio_button3.place(rely=0.55,relx=0.33)
        self.radiobox_var = tk.BooleanVar()
        self.radio_button4 = ctk.CTkRadioButton(self, text="Fill with certain value", variable=self.radio_var, value="4",command=self.toggle_radio_buttons_2)
        self.radio_button4.place(rely=0.6,relx=0.33)
        self.radio_var_2 = tk.StringVar()
        self.radio_button5 = ctk.CTkRadioButton(self, text="As String", variable=self.radio_var_2, value="5",command=self.SN)
        self.radio_button5.place(rely=0.7,relx=0.33)
        self.radio_button6 = ctk.CTkRadioButton(self, text="As Number", variable=self.radio_var_2, value="6",command=self.SN)
        self.radio_button6.place(rely=0.65,relx=0.33)
        self.textbox_2 = ctk.CTkEntry(self, placeholder_text="Enter The Value",width=160)
        self.textbox_2.place(rely=0.75,relx=0.33)
        self.menu_button_4 = ctk.CTkButton(self, text="↑", command=lambda stt = "null" : self.show_menu(stt),width=20,fg_color="black")
        self.menu_button_4.place(rely=0.525,relx=0.3)
        self.menu_button_4.configure(state=tk.DISABLED)
        self.radio_button1.configure(state=tk.DISABLED)
        self.radio_button2.configure(state=tk.DISABLED)
        self.radio_button3.configure(state=tk.DISABLED)
        self.radio_button4.configure(state=tk.DISABLED)
        self.radio_button5.configure(state=tk.DISABLED)
        self.radio_button6.configure(state=tk.DISABLED)
        self.textbox_2.configure(state=tk.DISABLED)
        #----------------------------------------------

        #---------------------Encoding-----------------
        self.cat_cols=[]
        self.option_buttons = []
        self.selected_options = set()
        self.selected_options_One = set()
        self.checkbox_var_3 = tk.BooleanVar()
        self.option8 = ctk.CTkCheckBox(self, text="Encoding",variable=self.checkbox_var_3, command=self.Update_Encoding)
        self.option8.place(rely=0.8,relx=0.01)
        self.option8.configure(state=tk.DISABLED)
        self.radio_var_3 = tk.StringVar()
                ####-----------Label Encoding-----------
        self.radio_button7 = ctk.CTkRadioButton(self, text="Label Encoding",variable=self.radio_var_3, value="Lab" ,command=self.LO)
        self.radio_button7.place(rely=0.85,relx=0.01)
        self.menu_button_1 = ctk.CTkButton(self, text="Select Columns", command=lambda stt = "Label Encoding" : self.show_menu(stt))
        self.menu_button_1.place(rely=0.85,relx=0.16)
        self.radio_button7.configure(state=tk.DISABLED)
        self.menu_button_1.configure(state=tk.DISABLED)
                ####-----------------------------------
                ####-----------One Hot Encoding--------
        self.radio_button8 = ctk.CTkRadioButton(self, text="One Hot Encoding", variable=self.radio_var_3, value="One",command=self.LO)
        self.radio_button8.place(rely=0.9,relx=0.01)
        self.menu_button_2 = ctk.CTkButton(self, text="Select Columns", command=lambda stt = "One Hot Encoding" : self.show_menu(stt))
        self.menu_button_2.place(rely=0.9,relx=0.16)
        self.radio_button8.configure(state=tk.DISABLED)
        self.menu_button_2.configure(state=tk.DISABLED)
                ####-----------------------------------
        #----------------------------------------------


        #--------------------Plot----------------------
        self.column_var = tk.StringVar()
        self.column_menu = ctk.CTkOptionMenu(self, variable=self.column_var, values=["Histogram","Box Plot","Scatter Plot","Bar Plot","Line Plot","Pie Chart","Heat Map"],command=self.toggle_6)
        self.column_menu.place(rely=0.8,relx=0.33)
        self.column_var.set("Select Plot")
        self.column_var_2 = tk.StringVar()
        self.column_menu_2 = ctk.CTkOptionMenu(self, variable=self.column_var_2, values=["None"],command=self.toggle_6)
        self.column_menu_2.place(rely=0.85,relx=0.33)
        self.column_var_2.set("Select Column")
        self.column_var_3 = tk.StringVar()
        self.column_menu_3 = ctk.CTkOptionMenu(self, variable=self.column_var_3, values=["None"],command=self.toggle_6)
        self.column_menu_3.place(rely=0.9,relx=0.33)
        self.column_var_3.set("Select Column")
        self.column_menu.configure(state=tk.DISABLED)
        self.column_menu_2.configure(state=tk.DISABLED)
        self.column_menu_3.configure(state=tk.DISABLED)
        self.checkbox_var_4 = tk.BooleanVar()
        self.option9 = ctk.CTkCheckBox(self, text="Plot",variable=self.checkbox_var_4, command=self.toggle_5,width=50)
        self.option9.place(rely=0.8,relx=0.25)
        self.option9.configure(state=tk.DISABLED)

    #-------------Show Columns To Select--------------
    def show_menu(self,ty):
        self.menu_window = tk.Toplevel(self)
        self.menu_window.title("Select Columns")
        self.selected_options.clear()
        self.selected_options_Drop.clear()
        self.selected_options_Null.clear()
        self.selected_options_One.clear()
        try :
            cols = self.data.columns.to_list()
            if ty == "Label Encoding" or ty=="One Hot Encoding":
                cols =self.data.select_dtypes(include=['object', 'bool']).columns.tolist()
            if ty == "null" and self.radio_var.get()=="median":
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                cols = self.data.select_dtypes(include=numerics).columns.tolist()
            for option in cols:
                var = tk.BooleanVar()
                cb = ctk.CTkCheckBox(self.menu_window, text=option, variable=var, command=lambda v=var, o=option , ty=ty : self.update_selection(v, o, ty))
                cb.pack(anchor="w")
                self.option_buttons.append((cb, var))
        except:
            pass
        close_button = ctk.CTkButton(self.menu_window, text="Close", command=lambda ty=ty :self.update_label(ty))
        close_button.pack(pady=10)
    #------------------------------------------------


    def update_selection(self, var, option,ty):
        if ty=="Label Encoding":
            if var.get():
                self.selected_options.add(option)
            else:
                self.selected_options.discard(option)
        elif ty == "One Hot Encoding":
            if var.get():
                self.selected_options_One.add(option)
            else:
                self.selected_options_One.discard(option)
        elif ty == "Drop":
            if var.get():
                self.selected_options_Drop.add(option)
            else:
                self.selected_options_Drop.discard(option)
        elif ty == "null":
            if var.get():
                self.selected_options_Null.add(option)
            else:
                self.selected_options_Null.discard(option)

    def update_label(self,ty):
        self.menu_window.destroy()
        if ty=="Label Encoding":
            selected_text = ", ".join(self.selected_options)+" SELECTED" if self.selected_options else "None"
        elif ty == "One Hot Encoding":
            selected_text = ", ".join(self.selected_options_One)+" SELECTED" if self.selected_options_One else "None"
        elif ty == "Drop" :
            selected_text = ", ".join(self.selected_options_Drop)+" SELECTED" if self.selected_options_Drop else "None"
        elif ty == "null":
            selected_text = ", ".join(self.selected_options_Null)+" SELECTED" if self.selected_options_Null else "None"
        messagebox.showinfo(self,f"{selected_text}")
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.insert ("1.0", f"{selected_text} for {ty}\n" )
        self.text_area.configure(state=tk.DISABLED)

    #------------------Drop Function -------------------
    def Drop_Func(self):
        self.preprocess_options["Drop"] = self.option_drop.get()
        if self.option_drop.get():
            self.menu_button_3.configure(state=tk.NORMAL)
        else:
            self.menu_button_3.configure(state=tk.DISABLED)
    #---------------------------------------------------

    
    def SN(self):
        self.preprocess_options["Number Or String"] = self.radio_var_2.get()

    def LO(self):
        self.preprocess_options["Encoding_type"] = self.radio_var_3.get()
        if self.radio_var_3.get()=="Lab":
            self.menu_button_1.configure(state=tk.NORMAL)
        else:
            self.menu_button_1.configure(state=tk.DISABLED)
        if self.radio_var_3.get()=="One":
            self.menu_button_2.configure(state=tk.NORMAL)
        else:
            self.menu_button_2.configure(state=tk.DISABLED)



    def browse(self):
        self.filename = filedialog.askopenfilename()
        name = self.filename.split('/')[-1]
        if self.filename:
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", f"Loaded file: {name}\n")
            self.text_area.configure(state=tk.DISABLED)

    def toggle_6(self,*args):
        self.preprocess_options["Choise_1"] = self.column_menu.get()
        self.preprocess_options["Choise_2"] = self.column_menu_2.get()
        self.preprocess_options["Choise_3"] = self.column_menu_3.get()
        if (self.column_var.get()=="Heat Map"):
            self.column_menu_2.configure(state=tk.DISABLED)
        else:
            self.column_menu_2.configure(state=tk.NORMAL)

        if (self.column_var.get()=="Bar Plot" or self.column_var.get()=="Line Plot" or self.column_var.get()=="Scatter Plot") and self.checkbox_var_4.get():
            self.column_menu_3.configure(state=tk.NORMAL)
        else:
            self.column_menu_3.configure(state=tk.DISABLED)
        if (self.column_var.get()=="Box Plot"):
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            cols = ["None"]+ self.data.select_dtypes(include=numerics).columns.tolist()
            self.column_menu_2.configure(values=cols)
        else:
            values_list= ["None"]+self.data.columns.to_list()
            self.column_menu_2.configure(values=values_list)


    def toggle_5(self):
        self.preprocess_options["Plot"] = self.option9.get()
        if self.option9.get():
            self.column_menu.configure(state=tk.NORMAL)
            self.column_menu_2.configure(state=tk.NORMAL)
            
        else:
            self.column_menu.configure(state=tk.DISABLED)
            self.column_menu_2.configure(state=tk.DISABLED)
            self.column_menu_3.configure(state=tk.DISABLED)



    def Random_Rows(self):
        self.preprocess_options["Show Random Rows"] = self.option1.get()
        if self.option1.get():
            self.textbox.configure(state=tk.NORMAL)
        else:
            self.textbox.configure(state=tk.DISABLED)

    def Update_Encoding(self):
        self.preprocess_options["Encoding"]= self.checkbox_var_3.get()
        if self.option8.get():
            self.radio_button7.configure(state=tk.NORMAL)
            self.radio_button8.configure(state=tk.NORMAL)
        else:
            self.radio_button7.configure(state=tk.DISABLED)
            self.radio_button8.configure(state=tk.DISABLED)
            self.menu_button_1.configure(state=tk.DISABLED)
            self.menu_button_2.configure(state=tk.DISABLED)
            self.radio_var_3.set("")

    def toggle_radio_buttons_2(self):
        self.preprocess_options["Method"] = self.radio_var.get()
        self.selected_options_Null.clear()
        if self.radio_var.get()=="4":
            self.radio_button5.configure(state=tk.NORMAL)
            self.radio_button6.configure(state=tk.NORMAL)
            self.textbox_2.configure(state=tk.NORMAL)
            self.radio_var_2.set("6")
        else:
            self.radio_button5.configure(state=tk.DISABLED)
            self.radio_button6.configure(state=tk.DISABLED)
            self.radio_var_2.set("")
            self.textbox_2.configure(state=tk.DISABLED)

    def toggle_radio_buttons(self):
        self.preprocess_options["Deal with Nulls"] = self.option10.get()
        if self.option10.get():
            self.radio_button1.configure(state=tk.NORMAL)
            self.radio_button2.configure(state=tk.NORMAL)
            self.radio_button3.configure(state=tk.NORMAL)
            self.radio_button4.configure(state=tk.NORMAL)
            self.menu_button_4.configure(state=tk.NORMAL)
            # self.radio_var.set("remove")
        else:
            self.radio_button1.configure(state=tk.DISABLED)
            self.radio_button2.configure(state=tk.DISABLED)
            self.radio_button3.configure(state=tk.DISABLED)
            self.radio_button4.configure(state=tk.DISABLED)
            self.radio_button5.configure(state=tk.DISABLED)
            self.radio_button6.configure(state=tk.DISABLED)
            self.menu_button_4.configure(state=tk.DISABLED)
            self.radio_var.set("")
            self.radio_var_2.set("")


    def update_options(self):
        self.preprocess_options["Descripe Numerical Columns"] = self.option3.get()
        self.preprocess_options["Descripe Categorical Columns"] = self.option4.get()
        self.preprocess_options["Show 10 tail rows"] = self.option5.get()
        self.preprocess_options["Show 10 head rows"] = self.option6.get()
        self.preprocess_options["Scaling"] = self.option7.get()
        self.preprocess_options["info"] = self.option2.get()
        

    #-----------------Reading and Loading Files------------------------
    def load_data(self):
        if not self.filename:
            messagebox.showerror("Error", "No file selected!")
            return
        global exten
        exten = self.filename.split('/')[-1].split('.')[-1]
        exten_list =['csv','fwf','json','html','xml','hdf','sql','xlsx','xls']
        if exten in exten_list:
            if exten == "csv":
                self.data = pd.read_csv(self.filename)
            elif exten == "fwf":
                self.data = pd.read_fwf(self.filename)
            elif exten == "json":
                self.data = pd.read_json(self.filename)
            elif exten == "html":
                self.data = pd.read_html(self.filename)
            elif exten == "xml":
                self.data = pd.read_xml(self.filename)
            elif exten == "clipboard":
                self.data = pd.read_clipboard(self.filename)
            elif exten == "xlsx":
                self.data = pd.read_excel(self.filename)
            elif exten == "hdf":
                self.data = pd.read_hdf(self.filename)
            elif exten == "sql":
                self.data = pd.read_sql(self.filename)
            elif exten == "xlsx":
                self.data = pd.read_excel(self.filename)
            elif exten == "xls":
                self.data = pd.read_excel(self.filename)
        else:
            messagebox.showerror("Error", f"Wrong Extension -->> {exten}, data must be one of this list {exten_list}")
            return
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.insert("1.0", "Data Loaded Successfully.\n")
        self.text_area.configure(state=tk.DISABLED)
        self.option1.configure(state=tk.NORMAL)
        self.option2.configure(state=tk.NORMAL)
        self.option_drop.configure(state=tk.NORMAL)
        self.option3.configure(state=tk.NORMAL)
        self.option4.configure(state=tk.NORMAL)
        self.option5.configure(state=tk.NORMAL)
        self.option6.configure(state=tk.NORMAL)
        self.option7.configure(state=tk.NORMAL)
        self.option8.configure(state=tk.NORMAL)
        self.option9.configure(state=tk.NORMAL)
        self.option10.configure(state=tk.NORMAL)
        self.process_button.configure(state=tk.NORMAL)
        self.save_button.configure(state=tk.NORMAL)
        self.cat_cols = self.data.select_dtypes(include=['object', 'bool']).columns.tolist()
        values_list= ["None"]+self.data.columns.to_list()
        self.column_menu_2.configure(values=values_list)
        self.column_menu_3.configure(values=values_list)
    #-------------------------------------------------------------------------------

    #------------Save------------------------
    def save_file(self):
        file_path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV files","*.csv"),("Excel files","*.xlsx"),("fwf files","*.fwf"),("XML files","*.xml"),("HTML files","*.html"),("JSON files","*.json"),("HDF5 files","*.hdf"),("SQL files","*.sql"),("All files","*.*")])
        if file_path.endswith('.csv'):
            self.data.to_csv(file_path,index=False)
        elif file_path.endswith('.json'):
            self.data.to_json(file_path,index=False)
        elif file_path.endswith('.html'):
            self.data.to_html(file_path,index=False)
        elif file_path.endswith('.xml'):
            self.data.to_xml(file_path,index=False)
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.insert("1.0", f"File Saved!\n")
        self.text_area.configure(state=tk.DISABLED)
    #-----------------------------------------
    def process_data(self):
        if self.preprocess_options["Descripe Categorical Columns"]:
            show(self.data.describe(include = ['object','bool'])) 
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", "Descripe Categorical Columns.\n")
            self.text_area.configure(state=tk.DISABLED)

        if self.preprocess_options["Descripe Numerical Columns"]:
            show(self.data.describe()) 
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", "Descripe Numerical Columns.\n")
            self.text_area.configure(state=tk.DISABLED)
#                 "Plot":False,
#                 "Choise_1":"",
#                 "Choise_2":"",
#                 "Choise_3":""
    #-----------------------NULLS---------------------------------------
        #---------------------Remove Nulls------------------
        if self.preprocess_options["Deal with Nulls"] and self.preprocess_options["Method"] == "remove":
            cols = list(self.selected_options_Null)
            if (len(cols))>0:
                self.data.dropna(subset=cols,inplace=True)
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", f"Nulls have been removed. in {self.selected_options_Null}\n")
            self.text_area.configure(state=tk.DISABLED)
        #-----------------------------------------------------
        #---------------------Medain--------------------------
        if self.preprocess_options["Deal with Nulls"] and self.preprocess_options["Method"] == "median":
            cols = list(self.selected_options_Null)
            for col in cols :
                med = self.data[col].median()
                self.data[col].fillna(med,inplace=True)

            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", f"Nulls have been replaced with median value in {self.selected_options_Null}.\n")
            self.text_area.configure(state=tk.DISABLED)
        #-----------------------------------------------------
        #---------------------FFILL---------------------------
        if self.preprocess_options["Deal with Nulls"] and self.preprocess_options["Method"] == "ffill":
            cols = list(self.selected_options_Null)
            self.data.loc[:,cols] = self.data.loc[:,cols].ffill()
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", f"Nulls have been replaced with forward. in {self.selected_options_Null}\n")
            self.text_area.configure(state=tk.DISABLED)
        #-----------------------------------------------------

        #---------------------Certain Value-------------------
        if self.preprocess_options["Deal with Nulls"] and self.preprocess_options["Method"] == "4":
            cols = list(self.selected_options_Null)
            var = self.textbox_2.get()
            if self.preprocess_options["Number Or String"] == "6" and var:
                try:
                    var=int(var)
                except ValueError :
                    messagebox.showerror("Error", "That is not a Number For Nulls!")
                    return
            else:
                var = str(var)
            if var :
                pass
            else:
                var=0
            # self.data[cols].fillna(var,inplace=True)
            self.data[cols] = self.data[cols].fillna(value=var)
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", f"Nulls have been Replaced With {var} as {type(var).__name__} in {self.selected_options_Null}.\n")
            self.text_area.configure(state=tk.DISABLED)
        #-----------------------------------------------------

    #-------------------------------------------------------------------


        #---------------------Label Encoding------------------
        if self.preprocess_options["Encoding"] and self.preprocess_options["Encoding_type"] == "Lab":
            labelencoder = LabelEncoder()
            cols= list(self.selected_options)
            self.selected_options.clear()
            self.data[cols] = self.data[cols].apply(labelencoder.fit_transform)
            self.cat_cols = self.data.select_dtypes(include=['object', 'bool']).columns.tolist()
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", f"{cols} Encoded (Label Encoding).\n")
            self.text_area.configure(state=tk.DISABLED)
        #-----------------------------------------------------

        #----------------------One Hot Encoding---------------
        if self.preprocess_options["Encoding"] and self.preprocess_options["Encoding_type"] == "One":
            cols= list(self.selected_options_One)
            self.selected_options_One.clear()
            self.data = pd.get_dummies(self.data, columns=cols, prefix=cols,dtype=int)
            self.cat_cols = self.data.select_dtypes(include=['object', 'bool']).columns.tolist()
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", f"{cols} Encoded (One Hot Encoding).\n")
            self.text_area.configure(state=tk.DISABLED)
        #-----------------------------------------------------

        #-------------------Tail------------------------------
        if self.preprocess_options["Show 10 tail rows"]:
            show(self.data.tail(10)) 
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", "Showed last 10 Rows.\n")
            self.text_area.configure(state=tk.DISABLED)
        #-----------------------------------------------------

        #--------------------Head-----------------------------
        if self.preprocess_options["Show 10 head rows"]:
            show(self.data.head(10)) 
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", "Showed First 10 Rows.\n")
            self.text_area.configure(state=tk.DISABLED)
        #-----------------------------------------------------

        #----------------------Scaling------------------------
        if self.preprocess_options["Scaling"]:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            num_cols = self.data.select_dtypes(include=numerics).columns.tolist()
            std_scaler = StandardScaler()
            self.data[num_cols] = std_scaler.fit_transform(self.data[num_cols])
            show(self.data.describe()) 
            self.text_area.insert("1.0", "Data is Scaled and described.\n")
        #----------------------------------------------------

        #-----------------------Info-------------------------
        if self.preprocess_options["info"]:
            buf = StringIO()
            self.data.info(buf=buf)
            info_str = buf.getvalue()
            info_lines = info_str.split('\n')
            column_info = info_lines[5:-2]  # Extract column information lines
            temp = []
            for line in column_info:
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 2:
                    column_name = parts[1]
                    column_type = parts[3]
                    column_Count = parts[2]
                    temp.append({'Column': column_name, 'Data Type': column_type, 'Count': column_Count})
            # Convert to DataFrame
            info_df = pd.DataFrame(temp)
            show(info_df)
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", "Showed info of data.\n")
            self.text_area.configure(state=tk.DISABLED)
        #-----------------------------------------------------

        #-----------------------Random Samples----------------
        if self.preprocess_options["Show Random Rows"]:
            num = self.textbox.get()
            try :
                num = int(num)
            except ValueError :
                self.text_area.configure(state=tk.NORMAL)
                self.text_area.insert("1.0", f"Enter a Number for Random Rows.\n")
                self.text_area.configure(state=tk.DISABLED)
                return
            num = abs(num)
            show(self.data.sample(num)) 
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", f"Showed Random {num} Rows.\n")
            self.text_area.configure(state=tk.DISABLED)
        #---------------------------------------------------

        #------------Drop----------------------
        if self.preprocess_options["Drop"]:
            cols = list(self.selected_options_Drop)
            self.selected_options_Drop.clear()
            self.data.drop(cols,axis=1,inplace=True)
            self.text_area.configure(state=tk.NORMAL)
            self.text_area.insert("1.0", f"{cols} Dropped.\n")
            self.text_area.configure(state=tk.DISABLED)
        #----------------------------------------

        
        #------------Plot------------------------
        if self.preprocess_options["Plot"]:
            type_plot = self.column_menu.get()
            col_1 = self.column_menu_2.get()
            col_2 = self.column_menu_3.get()
            if type_plot == "Select Plot":
                messagebox.showerror("Error", "Choose Plot!")
                return
            if (col_1=="None" or col_1 == "Select Column") and (type_plot != "Heat Map"):
                messagebox.showerror("Error", "Choose Column to Plot!")
                return
            plt.figure(figsize=(10, 5))
            if type_plot == "Bar Plot" and (col_2 == "None" or col_2 == "Select Column"):
                plt.bar(self.data[col_1],height=100)
            elif type_plot == "Bar Plot":
                plt.bar(self.data[col_1],self.data[col_2])
            elif type_plot== "Line Plot" and (col_2 == "None" or col_2 == "Select Column"):
                plt.plot(self.data[col_1])
            elif type_plot == "Line Plot":
                plt.plot(self.data[col_1],self.data[col_2])
            elif type_plot == "Histogram":
                plt.hist(self.data[col_1],bins=20)
            elif type_plot == "Scatter Plot":
                if (col_2=="None" or col_2 == "Select Column"):
                    messagebox.showerror("Error", "Choose 2nd Column to Plot!")
                    return
                plt.scatter(self.data[col_1],self.data[col_2])
            elif type_plot == "Pie Chart":
                (self.data[col_1].value_counts()*100.0 /len(self.data)).plot.pie(autopct='%.2f%%')
            elif type_plot == "Box Plot":
                try:
                    plt.boxplot(self.data[col_1])
                except :
                    messagebox.showerror("Error", "Check Column!")
                    return
            elif type_plot == "Heat Map":
                messagebox.showinfo(self,"Correlation between Numerical Columns")
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                cols =self.data.select_dtypes(include=numerics).columns.tolist()
                corr=self.data[cols].corr()
                sns.heatmap(corr,annot=True,cmap="coolwarm",linewidths=0.5)
            if (self.column_var.get()=="Bar Plot" or self.column_var.get()=="Line Plot" or self.column_var.get()=="Scatter Plot") and self.checkbox_var_4.get():
                if (col_2=="None" or col_2 == "Select Column"):
                    pass
                else:
                    plt.ylabel(f"{col_2}")
            plt.xlabel(f"{col_1}")
            plt.title(f"{type_plot}")
            plt.show()
        #----------------------------------------


        self.text_area.configure(state=tk.NORMAL)
        self.text_area.insert("1.0", "Data processed and visualized.\n")
        self.text_area.configure(state=tk.DISABLED)
if __name__ == "__main__":
    app = DataPreprocessorApp()
    app.mainloop()