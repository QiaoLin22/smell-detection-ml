import argparse
import sys
import pprint
import warnings
from algo import compute
import json
import glob
import os
import tkinter
from tkinter import messagebox

root = tkinter.Tk()
root.withdraw()
pp = pprint.PrettyPrinter(indent=4)
def train():
    envyres = compute("feature-envy.arff")
    datares = compute("data-class.arff")
    godres = compute("god-class.arff")
    longres = compute("long-method.arff")
    database = {**envyres, **datares, **godres, **longres}
    with open('database.json', 'w') as fp:
        json.dump(database, fp)

if glob.glob('database.json', recursive=True):
    with open('database.json') as f:
        database = json.load(f)
else:
    train()
    with open('database.json') as f:
        database = json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation',type = str,help='Choose your operation(listAll, modelAcc, compare,retrain)')
    parser.add_argument('--model',type=str, help='Choose the model used for the Training set(decision_tree, random_forest, naive_bayes, svc_linear, svc_poly, svc_rbf, svc_sigmoid)')
    parser.add_argument('--smell','--list', nargs='+', help='Choose the smells(feature-envy, data-class, god-class, long-method), could be multiple smells for modelAcc, only one for compare')
    args, unknown = parser.parse_known_args()
    sys.stdout.write(str(control(args,database)))

def control(args,database):
    smelllist = ['feature-envy','data-class','god-class','long-method']
    modellist = ['decision_tree','random_forest','naive_bayes','svc_linear','svc_poly','svc_rbf','svc_sigmoid']
    if args.operation == 'help':
        messagebox.showinfo("Information", "Welcome! \n Supported operations(listAll, run, compare,retrain). \r\n Supported models(decision_tree, random_forest, naive_bayes, svc_linear, svc_poly, svc_rbf, svc_sigmoid)."
                                           "\r\n Supported smells(feature-envy, data-class, god-class, long-method), smell input could be multiple for modelAcc, only one for compare. \r\n listAll list all available trained result,"
                                           "retrain discard previous training result and retrain. \r\n listAll and retrain don't take other iput. Run as 'python main.py listAll'. \r\n run and compare take two other inputs, model and smell,"
                                           "run take one model and multiple smell, compare take one model and only one smell.\r\n Run as ' python main.py compare --model decision_tree --smell feature-envy' or ' python main.py run --model naive_bayes --smell long-method god-class'.")
    if args.operation == 'retrain':
        if os.path.exists("database.json"):
            os.remove("database.json")
        train()
        with open('database.json') as f:
            database = json.load(f)
        listAll(database)
    if args.operation == 'listAll':
        listAll(database)
    if args.operation == 'run':
        list = args.smell
        for db in list:
            if db not in smelllist:
                messagebox.showwarning("Warning", 'your smell input contains non-existing smell')
        else:
            if args.model not in modellist:
                messagebox.showwarning("Warning", 'Input model not valid...')
            else:
                modelAcc(args.model,args.smell,database)
    if args.operation == 'compare':
        list = args.smell
        if(len(list) > 1):
            messagebox.showwarning("Warning", "Compare can only have 1 smell input")
        else:
            smell = list[0]
            if args.model not in modellist:
                messagebox.showwarning("Warning",'Input model not valid...')
            if smell not in smelllist:
                messagebox.showwarning("Warning", 'there is no file associated with this smell')
            else:
                compare(args.model, args.smell,database)


def listAll(database):
    pp.pprint(database)


def modelAcc(model, dblist,database):
    res = {}
    for db in dblist:
        res[db] = []
        for key in database:
            if db in key and model in key and 'test' in key:
                res[db].append(database[key])
    for key in res:
        if(len(res[key]) > 0):
          print(key + "  Acc: " + str(res[key][0]*100) + "  F1: " + str(res[key][1]*100))


def compare(model,dblist,database):
    res = {}
    for db in dblist:
        res['Training set'] = []
        res['Test set'] = []
        for key in database:
            if db in key and model in key and 'train' in key:
                res['Training set'].append(database[key])
            if db in key and model in key and 'test' in key:
                res['Test set'].append(database[key])
    for key in res:
       if(len(res[key]) > 0):
        print(key + "  Acc: " + str(res[key][0]*100) + "  F1: " + str(res[key][1]*100))


if __name__ == '__main__':
    main()