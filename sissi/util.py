#########################################################################
# borrowed from https://github.com/hexahedria/biaxial-rnn-music-composition/blob/master/
#########################################################################
import itertools

import midi
import numpy as np

lowerBound = 24
upperBound = 102


def startSentinel():
    def noteSentinel(note):
        position = note
        part_position = [position]

        pitchclass = (note + lowerBound) % 12
        part_pitchclass = [int(i == pitchclass) for i in range(12)]

        return part_position + part_pitchclass + [0] * 66 + [1]
    return [noteSentinel(note) for note in range(upperBound - lowerBound)]


def getOrDefault(l, i, d):
    try:
        return l[i]
    except IndexError:
        return d


def buildContext(state):
    context = [0] * 12
    for note, notestate in enumerate(state):
        if notestate[0] == 1:
            pitchclass = (note + lowerBound) % 12
            context[pitchclass] += 1
    return context


def buildBeat(time):
    return [2 * x - 1 for x in [time % 2, (time // 2) % 2, (time // 4) % 2, (time // 8) % 2]]


def noteInputForm(note, state, context, beat):
    position = note
    part_position = [position]

    pitchclass = (note + lowerBound) % 12
    part_pitchclass = [int(i == pitchclass) for i in range(12)]
    # Concatenate the note states for the previous vicinity
    part_prev_vicinity = list(itertools.chain.from_iterable(
        (getOrDefault(state, note + i, [0, 0]) for i in range(-12, 13))))

    part_context = context[pitchclass:] + context[:pitchclass]

    return part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0]


def noteStateSingleToInputForm(state, time):
    beat = buildBeat(time)
    context = buildContext(state)
    return [noteInputForm(note, state, context, beat) for note in range(len(state))]


def noteStateMatrixToInputForm(statematrix):
    # NOTE: May have to transpose this or transform it in some way to make Theano like it
    #[startSentinel()] +
    inputform = [noteStateSingleToInputForm(
        state, time) for time, state in enumerate(statematrix)]
    return inputform


def midiToNoteStateMatrix(midifile):

    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    span = upperBound - lowerBound
    time = 0

    state = [[0, 0] for x in range(span)]
    statematrix.append(state)
    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            statematrix.append(state)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - lowerBound] = [0, 0]
                        else:
                            state[evt.pitch - lowerBound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        return statematrix

                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix


def noteStateMatrixToMidi(statematrix, name="example"):
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = upperBound - lowerBound
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0, 0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time - lastcmdtime)
                                           * tickscale, pitch=note + lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time - lastcmdtime)
                                          * tickscale, velocity=40, pitch=note + lowerBound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)


if __name__ == "__main__":
    out = midiToNoteStateMatrix("./data/mozart/mz_545_3.mid")
    out = np.asarray(out)
    print(out.shape)
