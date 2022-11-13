import argparse
import json
import numpy as np

import mido


def notes2mid(notes):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    track.append(mido.MetaMessage('set_tempo', tempo=500000))
    track.append(mido.Message('program_change', program=0, time=0))

    previous_offset_time = 0
    cur_total_tick = 0

    for i in range(len(notes)):
        note = notes[i]
        if note[2] == 0:
            continue
        note[2] = int(round(note[2]))

        ticks_since_previous_onset = int(mido.second2tick(note[0], ticks_per_beat=480, tempo=500000))
        ticks_current_note = int(mido.second2tick(note[1], ticks_per_beat=480, tempo=500000))
        note_on_length = ticks_since_previous_onset - cur_total_tick
        note_off_length = ticks_current_note - note_on_length - cur_total_tick

        if note_off_length < 0 or note_on_length < 0:
            print (note_on_length, note_off_length, notes[i-1], note)

        if note_off_length < 20 and note_off_length > 0:
            print (note_on_length, note_off_length, note)

        track.append(mido.Message('note_on', note=int(note[2]), velocity=100, time=note_on_length))
        track.append(mido.Message('note_off', note=int(note[2]), velocity=100, time=note_off_length))
        cur_total_tick = cur_total_tick + note_on_length + note_off_length

    return mid


def main(args):
    with open(args.predicted_file) as json_data:
        tr = json.load(json_data)

    for i in tr.keys():
        to_convert = tr[i]
        mid = notes2mid(to_convert)
        # print ("song:", i)

    to_convert = tr[args.song_key]
    # print (len(to_convert))
    # print (to_convert)
    mid = notes2mid(to_convert)
    mid.save(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('predicted_file')
    parser.add_argument('song_key')
    parser.add_argument('output_path')

    args = parser.parse_args()

    main(args)
