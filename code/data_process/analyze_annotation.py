import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns. set_style('darkgrid')

def load_annotation(path = 'deuparl_validation/data/annotation.csv'):
    df = pd.read_csv(path)
    df['decade'] = [str(d[:3]) + '0s' for d in df['date']]
    #df['errors'] = [l.split(';') if isinstance(l, str) else None for l in df['errors']]
    #df['origination_of_errors'] = [l.split(';') if isinstance(l, str) else None for l in df['origination_of_errors']]
    try:
        df = df[['in_first_10', 'decade', 'text', 'is_sent', 'has_errors', 'errors', 'origination_of_errors', 'correction']]
    except:
        df = df[['decade', 'text', 'is_sent', 'has_errors', 'errors', 'origin_of_errors', 'correction']]
        df['errors'] = [None if pd.isna(row['errors']) else row['errors'].lower() for _, row in df.iterrows()]
    df = df.sort_values(by='decade')
    return df


def analyze(df):
    # how many are sentences?
    print(Counter(df['decade']))
    print(len(df))
    print(len(df[df.is_sent==False]))
    print(len(df[df.is_sent==True]))
    print(len(df[(df.is_sent==True) & (df.has_errors==True)]))
    print(len(df[(df.is_sent==True) & (df.has_errors==False)]))
    #raise ValueError
    tmp = defaultdict(list)
    tmp['decade'] = list(sorted(set(df['decade'])))

    #print(df[df.decade=='1870s'])
    #raise ValueError
    tmp = pd.DataFrame(tmp)

    tmp['non-sentence'] = df.groupby('decade').apply(lambda x: len(x[x.is_sent==False])/len(x) * 100).values
    tmp['perfect sentence'] = df.groupby('decade').apply(lambda x: len(x[(x.is_sent==True) & (x.has_errors==False)])/len(x) * 100).values
    tmp['sentence with errors'] = df.groupby('decade').apply(lambda x: len(x[(x.is_sent==True) & (x.has_errors==True)])/len(x) * 100).values
    #print(tmp)
    #raise ValueError
    tmp['sentence with errors'] = [x+y for x,y in zip(tmp['perfect sentence'], tmp['sentence with errors'])]
    tmp['non-sentence'] = [x+y for x,y in zip(tmp['non-sentence'], tmp['sentence with errors'])]
    tmp['#texts'] = df.groupby('decade').apply(lambda x: len(x)).values

    print(tmp)
    plt.figure(figsize=(6,5))
    #sns.barplot(data=tmp, x='decade', y='non-sentence', color='pink', label='non-sentence')
    ax = sns.barplot(data=tmp, x='decade', y='non-sentence', color='grey', label='non-sentence')
    #ax.bar_label(ax.containers[0])
    #sns.barplot(data=tmp, x='decade', y='sentence with errors', color='orange', label='sentence with issues')
    ax = sns.barplot(data=tmp, x='decade', y='sentence with errors', color='pink', label='sentence with issues')
    #ax.bar_label(ax.containers[0])
    #sns.barplot(data=tmp, x='decade', y='perfect sentence', color='lightblue', label='perfect sentence')
    ax = sns.barplot(data=tmp, x='decade', y='perfect sentence', color='lightblue', label='perfect sentence')
    #ax.bar_label(ax.containers[1])
    plt.yticks(range(0, 101, 10))
    plt.xticks(rotation=45)
    plt.ylabel('%', fontsize=12)
    plt.legend(loc='lower right')
    global path
    plt.tight_layout()
    plt.savefig(f"plots/{path.split('_')[0]}_is_sent_{len(df)}.png", dpi=300)
    plt.show()
    plt.close()

    #tmp = tmp[['decade', '#texts']]
    #tmp.to_csv('plots/errors/#texts.csv', index=False)

    df = df[(df.is_sent==True)]
    # check issues
    tmp = {'perfect sentence': len(df[df.has_errors==False])}
    errors = ['spelling', 'space', 'extra material', 'missing material', 'punctuation', 'symbol']
    #tmp = {'decade': ['all'], '#texts': [len(df)]}
    #tmp = {'decade': ['all'], '#texts': [len(df[df.has_errors == True])]}
    for error in errors:
        #tmp[error] = [len(df[(df.has_errors == True) & (df.errors.str.contains(error))]) / len(df[df.has_errors == True])]
        #tmp[error] = [len(df[(df.has_errors==True) & (df.errors.str.contains(error))])/len(df)]
        tmp[error] = len(df[(df.has_errors==True) & (df.errors.str.contains(error))])

    print(tmp)
    tmp = {k: v/len(df) * 100 for k, v in tmp.items()}
    tmp['punct & symbol'] = tmp['punctuation'] + tmp['symbol']
    del tmp['punctuation']
    del tmp['symbol']
    print(tmp)
    plt.figure(figsize=(3,4))
    plt.bar(x=range(len(tmp)), height=list(tmp.values()))
    labels = ['Perfect', 'Spelling', 'Space', 'Missing', 'Extra', 'Punct']
    #plt.xticks(range(len(tmp)), labels=list(tmp.keys()), rotation=45)
    plt.xticks(range(len(tmp)), labels=labels, rotation=45)
    plt.ylabel('%')
    plt.yticks(range(0, round(max(tmp.values()))+10, 10))
    plt.tight_layout()
    plt.savefig(f"plots/{path.split('_')[0]}_issues_{len(df)}.png", dpi=300)
    plt.show()
    #tmp = pd.DataFrame(tmp)
    #print(tmp)

    # check origins
    df = df[df.has_errors == True]
    tmp = {}
    origins = ['ocr', 'historic', 'genre', 'preprocessing']
    for origin in origins:
        tmp[origin] = len(df[df.origination_of_errors.str.contains(origin)])

    plt.figure(figsize=(2, 4))
    plt.bar(x=range(len(tmp)), height=list(tmp.values()))
    labels = ['OCR', 'Histo', 'Genre', 'Prep']
    # plt.xticks(range(len(tmp)), labels=list(tmp.keys()), rotation=45)
    plt.xticks(range(len(tmp)), labels=labels, rotation=45)
    plt.ylabel('%')
    plt.yticks(range(0, max(tmp.values())+10, 10))
    plt.tight_layout()
    plt.savefig(f"plots/{path.split('_')[0]}_origins_{len(df)}.png", dpi=300)
    plt.show()

    '''
    for decade, group in df.groupby('decade'):
        tmp['decade'].append(decade)
        tmp['#texts'].append(len(group))
        #tmp['#texts'].append(len(group[group.has_errors == True]))
        for error in errors:
            try:
                #tmp[error].append(len(group[(group.has_errors == True) & (group.errors.str.contains(error))]) / len(group[group.has_errors == True]))
                tmp[error].append(len(group[(group.has_errors == True) & (group.errors.str.contains(error))])/len(group))
            except:
                tmp[error].append(0)
    tmp = pd.DataFrame(tmp)
    print(tmp[tmp.decade=='all'][errors].values)
    plt.pie(tmp[tmp.decade=='all'][errors].values[0], labels=errors, autopct='%1.1f%%', startangle=140)
    plt.title('示例饼图')
    plt.show()

    #tmp.to_csv('deuparl_validation/error_dis_all.csv', index=False)
    #print(tmp)
    '''
    raise ValueError
    #print(set(df['has_errors']))  # ['errors'])


    tmp = {}
    errors = ['spelling', 'space', 'extra material', 'missing material', 'punctuation', 'symbol']
    #tmp = {'decade': ['all'], '#texts': [len(df)]}
    tmp = {'decade': ['all'], '#texts': [len(df[df.has_errors==True])]}
    for error in errors:
        tmp[error] = [len(df[(df.has_errors==True) & (df.errors.str.contains(error))])/len(df[df.has_errors==True])]
        #tmp[error] = [len(df[(df.has_errors==True) & (df.errors.str.contains(error))])/len(df)]
    for decade, group in df.groupby('decade'):
        tmp['decade'].append(decade)
        #tmp['#texts'].append(len(group))
        tmp['#texts'].append(len(group[group.has_errors==True]))
        for error in errors:
            try:
                tmp[error].append(len(group[(group.has_errors==True) & (group.errors.str.contains(error))])/len(group[group.has_errors==True]))
                #tmp[error].append(len(group[(group.has_errors==True) & (group.errors.str.contains(error))])/len(group))
            except:
                tmp[error].append(0)
    tmp = pd.DataFrame(tmp)
    tmp.to_csv('deuparl_validation/error_dis_error_sents_only.csv', index=False)
    print(tmp)
    print(set(df['has_errors']))#['errors'])


    tmp = {}
    errors = ['ocr', 'historic', 'genre', 'preprocessing']
    tmp = {'decade': ['all'], '#texts': [len(df)]}
    #tmp = {'decade': ['all'], '#texts': [len(df[df.has_errors == True])]}
    for error in errors:
        #tmp[error] = [len(df[(df.has_errors == True) & (df.origination_of_errors.str.contains(error))]) / len(df[df.has_errors == True])]
        tmp[error] = [len(df[(df.has_errors==True) & (df.origination_of_errors.str.contains(error))])/len(df)]
    for decade, group in df.groupby('decade'):
        tmp['decade'].append(decade)
        tmp['#texts'].append(len(group))
        for error in errors:
            try:
                #tmp[error].append(len(group[(group.has_errors == True) & (group.origination_of_errors.str.contains(error))]) / len(group[group.has_errors == True]))
                tmp[error].append(len(group[(group.has_errors==True) & (group.origination_of_errors.str.contains(error))])/len(group))
            except:
                tmp[error].append(0)
    tmp = pd.DataFrame(tmp)
    tmp.to_csv('deuparl_validation/originations_dis_all.csv', index=False)
    print(tmp)
    print(set(df['has_errors']))  # ['errors'])

    tmp = {}
    errors = ['ocr', 'historic', 'genre', 'preprocessing']
    #tmp = {'decade': ['all'], '#texts': [len(df)]}
    tmp = {'decade': ['all'], '#texts': [len(df[df.has_errors == True])]}
    for error in errors:
        tmp[error] = [len(df[(df.has_errors == True) & (df.origination_of_errors.str.contains(error))]) / len(df[df.has_errors == True])]
        #tmp[error] = [len(df[(df.has_errors == True) & (df.origination_of_errors.str.contains(error))]) / len(df)]
    for decade, group in df.groupby('decade'):
        tmp['decade'].append(decade)
        #tmp['#texts'].append(len(group))
        tmp['#texts'].append(len(group[group.has_errors==True]))
        for error in errors:
            try:
                tmp[error].append(len(group[(group.has_errors == True) & (group.origination_of_errors.str.contains(error))]) / len(group[group.has_errors == True]))
                #tmp[error].append(len(group[(group.has_errors == True) & (group.origination_of_errors.str.contains(error))]) / len(group))
            except:
                tmp[error].append(0)
    tmp = pd.DataFrame(tmp)
    tmp.to_csv('deuparl_validation/originations_dis_error_sents_only.csv', index=False)
    print(tmp)
    print(set(df['has_errors']))  # ['errors'])




    '''
    tmp = pd.DataFrame(tmp)
    tmp.rename(columns={0: 'error rate'}, inplace=True)
    #print(tmp.index)
    print(tmp.columns)

    #print(tmp)
    #print(tmp.index)
    #tmp['decade'] = tmp.index.tolist()
    tmp.reset_index(inplace=True)
    #tmp['error rate'] = tmp.values
    print(tmp)
    sns.lineplot(data=tmp, x='decade', y='error rate')
    #plt.figure()
    #plt.plot(x=range(len(tmp.index)), y=tmp.values)
    plt.show()
    #plt.figure()
    '''


if __name__ == '__main__':
    #path = 'data/annotations/deuparl_annotation - batch_1_human.csv'

    # German
    path = 'deuparl_validation/data/annotation.csv'
    df = load_annotation(path)
    df = df[df.in_first_10 == 1]

    # English
    '''
    path1 = 'hansard_validation/data/sents_hansard_Steffen - fix.csv'
    path2 = 'hansard_validation/data/sents_hansard_Yanran - annotation.csv'

    df1 = pd.read_csv(path1)

    df2 = pd.read_csv(path2)
    print(df2)
    df2 = df2[~df2.sent.isin(df1['sent'])]
    print(df2)

    df = pd.concat([df1, df2], ignore_index=True)
    print(df)

    df.to_csv('hansard_validation/data/annotation.csv', index=False)
    '''
    path = 'hansard_validation/data/annotation.csv'
    #df = pd.read_csv(path)
    #df.rename(columns={'sent': 'text'}, inplace=True)
    #print(df.columns)
   # df.to_csv(path, index=False)
   # raise ValueError
    df = load_annotation(path)


    analyze(df)
    #raise ValueError


    #df['decade'] = [str(d[:3])+'0s' for d in df['date']]
    #print(df)
    #analyze(df)
    #df_sent = df[(df.is_sent == True) & (df.has_errors == True)]
    #print(list(df_sent['correction']))
    #df_muss = df_sent[df_sent.correction.str.contains('muß')]#, na=False)]
    #df_muss = df[df.text.str.contains('muß') & df.correction.str.contains('muß')]
    #print(df_muss)
