import pandas as pd
import os
from termcolor import colored


class mergeall():
    def __init__(self,folderin,listfile=None,folderout=None,fileout=None,write=False):
        if os.path.isdir(folderin):
            self.folderin = folderin
        else:
            exit()
        self.listcentre = listfile
        if folderout != None:
            self.folderout = folderout
        else:
            self.folderout="./"

        if fileout != None:
            self.fileout = fileout
        else:
            self.fileout = "ListVoxel.csv" # Change default name ??
        self.writeflag = write

    def update(self):
        Merge= pd.DataFrame()
        i=0
        for centre in self.listcentre:
            filecentre=pd.read_excel(os.path.join(self.folderin,centre),index_col=0)
            centrename , centreextension = os.path.splitext(centre)
            Dcd=filecentre["Dcd"]
            for patient in Dcd.index:
                patientpath=os.path.join(self.folderin, "Results", centrename, patient)
                if os.path.isdir(patientpath):
                    print(colored("Patient n°: "+patient,"green"))
                    i=i+1
                    patientfile=os.path.join(patientpath, patient+".csv")
                    patientf=pd.read_csv(patientfile)
                    patientf.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
                    patientf["ID"]= patient
                    patientf["Dcd"]= Dcd[patient]
                    Merge=pd.concat([Merge,patientf],sort=False,ignore_index=True)

        Mergefile = Merge.set_index(["ID"])
        Mergefile.dropna(inplace=True)
        #print(colored("Total number of patients is: "+i, "red"))
        if self.writeflag:
            print(colored("Writing...", "magenta"))
            Mergefile.to_csv(os.path.join(self.folderout, self.fileout))
            print(colored("DONE!",  "red"))
        return Mergefile




