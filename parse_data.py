import pandas as pd


def write_tweets_to_txt(file):
    df = pd.read_csv(file, encoding='ISO-8859-1')
    column_data = df["Tweet"]
    row_count = 0

    with open('atheism_without_none.txt', 'w', encoding='ISO-8859-1') as f:
        for value in column_data:
            if(row_count % 43 == 0):
                f.write('********************************************' + '\n')
            f.write(str(value) + '\n')
            row_count += 1
            
write_tweets_to_txt('atheism_without_none.csv')