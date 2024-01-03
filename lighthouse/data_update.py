# load all .out from a folder

# create or update an existing dataframe

import pandas as pd
import os
import json

def load_existing_dataframe(existing_df_path):
    """Load an existing DataFrame from a CSV file."""
    if os.path.exists(existing_df_path):
        return pd.read_csv(existing_df_path)
    else:
        return pd.DataFrame()

def read_json_file(file_path):
    """Read a JSON file and convert it to a DataFrame."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    data = {key: [value] if not isinstance(value, list) else value for key, value in data.items()}
    return pd.DataFrame.from_dict(data)

def append_to_dataframe(main_df, new_df):
    """Append a new DataFrame to an existing DataFrame."""
    return pd.concat([main_df, new_df], ignore_index=True)

def process_json_files_in_folder(folder_path="input", existing_df_file='bulb.csv'):
    """Process all JSON files in a folder, appending to an existing DataFrame or creating a new one."""
    existing_df_path = os.path.join("output", existing_df_file)
    final_df = load_existing_dataframe(existing_df_path)

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            new_df = read_json_file(file_path)
            final_df = append_to_dataframe(final_df, new_df)

    return final_df.drop_duplicates(subset=["id"])

# TODO
# backups
# new labels?
if __name__=='__main__':
    final_df = process_json_files_in_folder()
    final_df.to_csv('output/bulb.csv', index=False)
    print(final_df)

    print(final_df.columns)
