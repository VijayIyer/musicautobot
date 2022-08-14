from musicautobot.multitask_transformer import *
from musicautobot.music_transformer import *
from musicautobot.config import *
import music21
import torch
import argparse
import os

DATA_PATH = os.path.join('data', 'numpy')

def load_model(model='multitask'):
    data = load_data(DATA_PATH, 'musicitem_data_save.pkl', num_workers=1)
    if model=='multitask':
        learn = multitask_model_learner(data, pretrained_path=os.path.join(DATA_PATH, 'pretrained', 'MultitaskSmall.pth'))
    else:
        learn =  music_model_learner(data, pretrained_path=os.path.join(DATA_PATH, 'pretrained', 'MusicTransformer.pth'))
    if torch.cuda.is_available(): learn.model.cuda()
    return learn

def generate_next_notes():
    '''
    given a melody, continue it
    '''
    learn = load_model('multitask')
    midi = args.file
    bpm = args.bpm
    temperatures = (float(args.noteTemp), float(args.durationTemp))
    top_k, top_p = (int(args.topK), float(args.topP))
    n_words = int(args.nSteps)

    full = nw_predict_from_midi(learn, midi=midi, n_words=n_words, temperatures=temperatures, seed_len=args.seed_len, 
                                         top_k=top_k, top_p=top_p)
    stream = full.to_stream(bpm=bpm)
    stream.write("midi", fp = args.out)

def generate_chords_for_melody():
    '''
    given a melody, harmonize it
    '''
    learn = load_model('multitask')
    midi = args.file
    bpm = args.bpm
    temperatures = (float(args.noteTemp), float(args.durationTemp))
    top_k, top_p = (int(args.topK), float(args.topP))
    n_words = int(args.nSteps)

    full = s2s_predict_from_midi(learn, midi=midi, n_words=n_words, temperatures=temperatures, seed_len=args.seed_len, 
                                         pred_melody=False, use_memory=True, top_k=top_k, top_p=top_p)
    stream = full.to_stream(bpm=bpm)
    stream.write("midi", fp = args.out)

def generate_melody_for_chords():
    '''
    given a harmony, generate melody for it
    '''
    learn = load_model('multitask')
    midi = args.file
    bpm = args.bpm
    temperatures = (float(args.noteTemp), float(args.durationTemp))
    top_k, top_p = (int(args.topK), float(args.topP))
    n_words = int(args.nSteps)

    full = s2s_predict_from_midi(learn, midi=midi, n_words=n_words, temperatures=temperatures, seed_len=args.seed_len, 
                                         pred_melody=True, use_memory=True, top_k=top_k, top_p=top_p)
    stream = full.to_stream(bpm=bpm)
    stream.write("midi", fp = args.out)

    
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default="in.mid")
    parser.add_argument("--task", default="generate melody")
    parser.add_argument("--out", default="out.mid")
    parser.add_argument("--seed_len", default=12)
    parser.add_argument("--noteTemp", default=1.2)
    parser.add_argument("--durationTemp", default=0.8)
    parser.add_argument("--topK", default = 20)
    parser.add_argument("--topP", default = 0.9)
    parser.add_argument("--nSteps", default = 200)
    parser.add_argument("--bpm", default = 120)
    args = parser.parse_args()
    if args.task == "generate next notes":
        generate_next_notes()
    elif args.task == "generate chords for melody":
        generate_chords_for_melody()
    elif args.task == "generate melody for chords":
        generate_melody_for_chords()
    
    else:
        print("pick one of the 4 options for task\n1. generate melody\n2.generate chords\n3.harmonize\n4.melodize")

