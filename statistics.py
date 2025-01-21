#!/usr/bin/env python
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def avg(v) :
    return sum(v)/len(v)

def stdev(v) :
    mean = avg(v)
    s = sum([(x-mean)**2 for x in v])
    return (s/len(v))**0.5

def median(v) :
    v1 = v
    v1.sort()
    if len(v1) % 2 == 0:
        return (v1[len(v1)//2]+v1[(len(v1)+1)//2])/2
    return v1[len(v1)//2]

def get_files_with_prefix(prefix):
    # Get the list of files in the current directory
    files = os.listdir('./data')
    
    # Filter files that start with the given prefix
    filtered_files = [file for file in files if file.startswith(prefix)]
    
    return filtered_files

def crop_last_percent(values, percent):
    v = values
    v.sort()
    if not 0 <= percent <= 100:
        print("error")
        return values
    n = int(len(v) * percent / 100)
    ret = v[:-n]
    return ret

def crop_first_percent(values, percent):
    v = values
    v.sort()
    if not 0 <= percent <= 100:
        print("error")
        return values
    n = int(len(v) * percent / 100)
    ret = v[n:]
    return ret

def crop_extremities_percent(values, percent):
    v = values
    v.sort()
    if not 0 <= percent <= 100:
        print("error")
        return values
    percent/2
    n = int(len(v) * percent / 100)
    ret = v[n:-n]
    return ret

def get_right_files(files_list, algo, dim, optimization, version, order) :
    v = [[y for y in x.split("_")] for x in files_list]
    v = [x for x in v if len(x) == 6]
    ans = []
    for i in range(len(v)) :
        if algo == "." or algo in v[i][1] :
            ans.append(i)
        if (dim != "." and dim != v[i][2]) and i in ans :
            ans.remove(i)
        if (optimization != "." and optimization not in v[i][3]) and i in ans :
            ans.remove(i)
        if (version != "." and version not in v[i][4]) and i in ans :
            ans.remove(i)
        if (order != "." and order not in v[i][5]) and i in ans :
            ans.remove(i)
    res = [files_list[i] for i in ans]
    return res

def basic_stats(fileName) :
    with open(fileName,"rb") as f :
        file = f.read().decode()
        v = []
        for x in file.split("\n") :
            try :
                if len(x) > 0 :
                    v.append(int(x))
            except :
                pass
    print(f"{fileName} {int(round(avg(v),0))}±{int(round(stdev(v),0))} median: {median(v)}")
    print(f"{int(round(avg(v),0))} & {int(round(stdev(v),0))} & {median(v)}")

def get_param(string) :
    v = string.split(",")
    return v[0],v[1],v[2],v[3],v[4]


def plot(data,graphType) :
    import plotly.express as px
    try :
        if "histogram" in graphType :
            fig = px.histogram(data,marginal="box",barmode='overlay')
        elif "box" in graphType :
            fig = px.box(data, points="all",log_y=True)
            #fig = px.box(data,log_y=True)
        elif "violin" in graphType :
            fig = px.violin(data,log_y=True)
        else :
            fig = px.histogram(data,marginal="box",barmode='overlay')
        fig.show()
        return
    except :
        print("errore")


import matplotlib.pyplot as plt
import numpy as np

def funzione():
    datas = {}
    algos = ["insertionSort", "selectionSort", "bubbleSort"]
    lengths = ["1000", "10000", "100000"]
    flags = ["O1", "O2"]
    versions = {
        "insertionSort": ["0", "2", "4"],
        "selectionSort": ["0", "1", "2", "3", "8"],
        "bubbleSort": ["0", "1", "8"]
    }

    for algo in algos:
        if algo not in datas:
            datas[algo] = {}
        for length in lengths:
            if length not in datas[algo]:
                datas[algo][length] = {}
            for flag in flags:
                if flag not in datas[algo][length]:
                    datas[algo][length][flag] = {}
                for version in versions[algo]:
                    file = f"data/tmp_{algo}_{length}_{flag}_v{version}_0"
                    try:
                        with open(file, "rb") as f:
                            a = f.read().decode()
                            v = []
                            for x in a.split("\n"):
                                if x.isdigit():  # Controlla che la riga sia un numero
                                    v.append(int(x))
                            if v:
                                median_value = np.median(v)
                                datas[algo][length][flag][version] = median_value
                    except FileNotFoundError:
                        print(f"file non trovato {file}")
                        pass

    
    algos = ["insertionSort", "selectionSort", "bubbleSort"]
    for algo in algos:
        plt.figure()
        for flag in ["O1", "O2"]:
            for version in versions[algo]:
                x = []
                y = []
                for length, flags in datas[algo].items():
                    if flag in flags and version in flags[flag]:
                        x.append(int(length))
                        y.append(flags[flag][version])
                if x and y:
                    plt.plot(x, y, marker='o', label=f"{algo} {flag} v{version}")
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Data Length')
        plt.ylabel('Median Time (ms)')
        plt.legend()
        plt.title(f'Algorithm Performance - {algo}')
        plt.show()












def myFunction() :
    datas = []
    algos = ["insertionSort","selectionSort","bubbleSort"]
    lenghts = ["1000","10000","100000"]
    flags = ["O1","O2"]
    for algo in algos :
        tmp1 = []
        for lenght in lenghts :
            tmp2 = []
            for flag in flags :
                tmp3 = []
                if "insertion" in algo :
                    for version in ["0","2","4"] :
                        file = "data/tmp_"+algo+"_"+lenght+"_"+flag+"_v"+version+"_0"
                        try :
                            with open(file,"rb") as f :
                                a = f.read().decode()
                                v = []
                                for x in a.split("\n") :
                                    if len(x) > 0 :
                                        v.append(int(x))
                            tmp3[version] = v
                        except :
                            print(f"file non trovato {file}")
                            pass
                    
                
    print(datas)

def main() :
    if len(sys.argv) < 2 :
        print("Usage:")
        print("statistics.py nameFIle\tto print average and standard deviation")
        print("statistics.py [histogram/box] algo,dim,optimization,version,order\tto print graphs ( . means any)")
    if len(sys.argv) == 2 :
        if "my" in sys.argv[1] :
             funzione()
        else :
            basic_stats(sys.argv[1])
        return
    # prendi tutti i file della cartella corrente che iniziano per sys.argv[1] ed estraine i dati, poi plotta i risultati raccolti
    if len(sys.argv) > 2 :
        graphType = sys.argv[1]
        filesToPrint = []
        files = get_files_with_prefix("tmp_")
        for i in range(2,len(sys.argv)) :
            algo, dim, optimization, version, order = get_param(sys.argv[i])
            filesToPrint = list(set(filesToPrint + get_right_files(files,algo, dim, optimization, version, order)))
        if len(filesToPrint) < 1 :
            print("no files found")
            return
        filesToPrint.sort()
        for file in filesToPrint :
            try :
                with open(f"data/{file}","rb") as f :
                    data = f.read().decode()
                    try :
                        v = [int(x) for x in data.split("\n") if len(x) > 0]
                    except :
                        print(f"eccezione nel trasformare i dati")
                        pass
                    v = crop_extremities_percent(v,6)
                    if len(v) > 0 :
                        try :
                            df = pd.concat([pd.DataFrame({file:v}),df], ignore_index=True)
                        except :
                            df = pd.DataFrame({file:v})
            except :
                print(f"eccezione ")
                print(data)
                pass
        try :
            plot(df,graphType)
        except :
            print("errore")
            pass

def example_plot() :
    df = px.data.tips()
    fig = px.histogram(df, x="total_bill", y="tip", color="sex",
                    marginal="violin", # or violin, rug
                    hover_data=df.columns)
    fig.update_layout(
        title = "Evolution of X"
    )
    fig.show()

if __name__ == "__main__" :
    main()