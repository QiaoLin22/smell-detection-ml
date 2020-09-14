import argparse
import sys
import pprint
import warnings
from algo import compute
import json
import glob
pp = pprint.PrettyPrinter(indent=4)

if glob.glob('database.json', recursive=True):
    with open('database.json') as f:
        database = json.load(f)
else:
    envyres = compute("feature-envy.arff")
    datares = compute("data-class.arff")
    godres = compute("god-class.arff")
    longres = compute("long-method.arff")
    database = {**envyres, **datares, **godres, **longres}
    with open('database.json', 'w') as fp:
        json.dump(database, fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation',type = str,help='Choose your operation(listAll, modelAcc, compare)')
    parser.add_argument('--model',type=str, help='Choose the model used for the Training set(decision_tree, random_forest, naive_bayes, svc_linear, svc_poly, svc_rbf, svc_sigmoid)')
    parser.add_argument('--smell','--list', nargs='+', help='Choose the smells(feature-envy, data-class, god-class, long-method), could be multiple smells for modelAcc, only one for compare')
    args = parser.parse_args()
    sys.stdout.write(str(control(args,database)))

def control(args,database):
    smelllist = ['feature-envy','data-class','god-class','long-method']
    modellist = ['decision_tree','random_forest','naive_bayes','svc_linear','svc_poly','svc_rbf','svc_sigmoid']
    if args.operation == 'listAll':
        listAll(database)
    if args.operation == 'run':
        modelAcc(args.model,args.smell,database)
    if args.operation == 'compare':
        list = args.smell
        if(len(list) > 1):
            warnings.warn("Compare can only have 1 smell input")
        else:
            smell = list[0]
            if args.model not in modellist:
                warnings.warn('Input model not valid...')
            if smell not in smelllist:
                warnings.warn('there is no file associated with this smell')
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