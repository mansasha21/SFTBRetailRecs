import re
import typing as tp

import numpy as np


def clean_first_step(string: str) -> str:
    r1: str = r"\d+(\.\d+)?[гслмк]?[рк]?[м]?[\%]?"
    r2: str = r"(?<!\S)./"
    r3: str = r" \/[^\/]*\/"
    r4: str = r"\w+\."
    r5: str = r"[а-яА-Я]\/"
    r6: str = r"\w+\([^\)]*\)"
    r7: str = r"\*"

    regex = re.compile(r"(%s|%s|%s|%s|%s|%s|%s)" % (r1, r2, r3, r4, r5, r6, r7), re.IGNORECASE)

    return re.sub(regex, '', string)


def clean_second_step(string: str) -> str:
    r1: str = r"\(\+\)"
    
    regex = re.compile(r"(%s)" % (r1), re.IGNORECASE)
    
    return re.sub(regex, '', string)


def specify_deduction(string: str) -> str:
    return string \
                .replace("Сиг-ты", "Сигареты") \
                .replace("К-са", "Колбаса")


def process_sentence(sent: str) -> str:
    return specify_deduction(clean_second_step(clean_first_step(sent)))


def get_sentence_embedding(string: str) -> np.array:
    sent: tp.List[np.array] = []
    
    for word in process_sentence(string).split():
        curr_emb: np.array = emb.get(word.lower())

        if curr_emb is not None: sent.append(curr_emb)
        else: sent.append(emb["<unk>"])
                          
    return np.mean(np.array(sent), axis=0)
