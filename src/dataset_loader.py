import pandas as pd
import numpy as np

def extract_block(content):
    if "blk_" in str(content):
        start = content.find("blk_")
        part = content[start:].split()[0].strip(',')
        return part
    return np.nan

def load_csv(path_logs, path_labels):
    logs = pd.read_csv(path_logs)
    labels = pd.read_csv(path_labels)

    logs['BlockId'] = logs['Content'].apply(extract_block)
    logs = logs.dropna(subset=["BlockId"])

    block_seqs = logs.groupby("BlockId")["EventTemplate"].apply(list)
    labels = labels.set_index("BlockId")["label"]
    label2id = {"Normal": 0, "Anomaly": 1}

    df = pd.DataFrame({
        "text": block_seqs,
        "label": labels
    }).dropna().reset_index(drop=True)

    df["label"] = df["label"].map(label2id).astype(int)

    df["text"] = df["text"].apply(
        lambda seq: ', '.join(seq) if isinstance(seq, (list, np.ndarray)) else str(seq)
    )
    return df


def get_events_dict(path):
    event_templates = pd.read_csv(path)
    return dict(zip(event_templates['EventId'], event_templates['EventTemplate']))

def load_npz(path_npz, path_event_templates=None):
    data = np.load(path_npz, allow_pickle=True)
    x_train = data['x_data']
    y_train = data['y_data']

    df = pd.DataFrame({
        'text': x_train,
        'label': y_train
    })
    if path_event_templates:
        eventid2event = get_events_dict(path_event_templates)
        df['text'] = df['text'].apply(lambda seq: ', '.join([eventid2event[event_id] for event_id in seq]))
    else:
        df['text'] = df['text'].apply(lambda seq: ', '.join(seq))

    return df