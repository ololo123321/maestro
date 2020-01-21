import os
import json
import random
import shutil
from collections import defaultdict
from typing import List, Union
from tqdm import tqdm
import pretty_midi
from pretty_midi import Note, ControlChange, Instrument, PrettyMIDI


class DataConverterV1:
    """
    Конвертация MIDI-файлов в токены событий и наоборот
    Педаль не учитывается
    """
    def __init__(self, step=10, max_delay=1000):
        self.step = step
        self.max_delay = max_delay
        self._num_bins_velocity = 32
        self._bin_size_velocity = 3

    def midi2tokens(self, midi: PrettyMIDI) -> List[str]:
        piano_midi = midi.instruments[0]
        notes = piano_midi.notes

        bin2events = defaultdict(list)
        sorted_notes = sorted(notes, key=lambda x: x.start)
        for note in sorted_notes:
            start_bin_id = note.start * 1000 // self.step
            end_bin_id = note.end * 1000 // self.step
            velocity = self._velocity2bin(note.velocity)
            bin2events[start_bin_id] += [f"SET_VELOCITY_{velocity}", f"NOTE_ON_{note.pitch}"]
            bin2events[end_bin_id].append(f"NOTE_OFF_{note.pitch}")

        shifts = list(range(self.step, self.max_delay + 1, self.step))

        def get_delays(start, end):
            res = []
            diff = end - start
            if diff < step:
                return res
            i = -1
            while True:
                shift_i = shifts[i]
                diff_new = diff - shift_i
                if diff_new < 0:
                    i -= 1
                elif diff_new < step:
                    res.append(shift_i)
                    return res
                else:
                    res.append(shift_i)
                    diff = diff_new

        sorted_items = sorted(bin2events.items(), key=lambda x: x[0])
        tokens = []
        for i in range(len(sorted_items) - 1):
            tokens += sorted_items[i][1]
            start = sorted_items[i][0] * self.step
            end = sorted_items[i + 1][0] * self.step
            delays = get_delays(start, end)
            tokens += sorted([f"TIME_SHIFT_{x}" for x in delays], reverse=True)
        return tokens

    def tokens2midi(self, tokens: List[str]) -> PrettyMIDI:
        midi = PrettyMIDI()
        program = instrument_name_to_program('Acoustic Grand Piano')
        piano = Instrument(program=program)
        velocity = 0
        t = 0
        pitch2start = {}
        pitch2end = {}
        pitch2velocity = {}
        n_tokens = len(tokens)
        for i in range(n_tokens):
            tok_i = tokens[i]
            value = int(tok_i.split("_")[-1])
            if tok_i.startswith("SET_VELOCITY"):
                velocity = value
            elif tok_i.startswith("TIME_SHIFT"):
                t += value
                pitch2end = {k: v + value for k, v in pitch2end.items()}
            elif tok_i.startswith("NOTE_ON"):
                pitch2start[value] = t
                pitch2end[value] = t
                pitch2velocity[value] = velocity
            elif tok_i.startswith("NOTE_OFF"):
                if value in pitch2start:
                    start = pitch2start.pop(value)
                    end = pitch2end.pop(value)
                    if end > start:
                        note = Note(
                            velocity=self._bin2velocity(pitch2velocity.pop(value)),
                            pitch=value,
                            start=start / 1000,
                            end=end / 1000
                        )
                        piano.notes.append(note)
        midi.instruments.append(piano)
        return midi

    def _velocity2bin(self, velocity):
        return min(self._num_bins_velocity - 1, velocity // self._bin_size_velocity)

    def _bin2velocity(self, velocity):
        return velocity * self._bin_size_velocity


class DataConverterV2(DataConverterV1):
    """
    Конвертация MIDI-файлов в токены событий и наоборот
    Педаль не учитывается
    """
    def __init__(self, step=10, max_delay=1000):
        super().__init__(step, max_delay)
        self._sustain_pedal_number = 64
        self._sustain_pedal_thresh = 64

    def midi2tokens(self, midi):
        """
        Педаль нажата, если cc.number == 64 and cc.value >= 64
        Педаль отпущена, если cc.number == 64 and cc.value < 64
        Если во время нажатой педали произошло окончание ноты,
        то оно откладывется до наступления одного из двух событий:
        1. Начало той же ноты
        2. Окончание педали
        """
        piano = midi.instruments[0]
        notes = piano.notes
        control_changes = [x for x in piano.control_changes if x.number == self._sustain_pedal_number]
        events = sorted(notes + control_changes, key=lambda x: x.start if isinstance(x, Note) else x.time)

        pairs = set()
        eps = 1e-6
        is_pedal_on = False
        n_events = len(events)

        to_close_with_pedal = []

        for i, event in enumerate(events):
            if isinstance(event, Note):
                t = event.start
                velocity = self._velocity2bin(event.velocity)
                pitch = event.pitch
                pairs.add((t, f"SET_VELOCITY_{velocity}", f"NOTE_ON_{pitch}"))
                if is_pedal_on:
                    if i != n_events - 1:
                        for event_j in events[i + 1:]:
                            if isinstance(event_j, Note):
                                if event_j.pitch == pitch:
                                    pairs.add((event_j.start - eps, f"NOTE_OFF_{pitch}"))
                                    break
                            else:
                                if event_j.value < self._sustain_pedal_thresh:
                                    to_close_with_pedal.append(pitch)
                                    break
                else:
                    pairs.add((event.end, f"NOTE_OFF_{pitch}"))
            else:
                t = event.time
                if event.value >= self._sustain_pedal_thresh:
                    if not is_pedal_on:
                        is_pedal_on = True
                        pairs.add((t, "PEDAL_ON"))
                else:
                    if is_pedal_on:
                        is_pedal_on = False
                        group = [t, "PEDAL_OFF"] + [f"NOTE_OFF_{pitch}" for pitch in to_close_with_pedal]
                        pairs.add(tuple(group))
                        to_close_with_pedal.clear()

        pairs = sorted(pairs, key=lambda x: x[0])  # сортировка сначала по времени, а затем по названию
        bin2events = defaultdict(list)
        for group in pairs:
            t = group[0]
            bin_id = t * 1000 // self.step
            bin2events[bin_id] += list(group[1:])

        shifts = list(range(self.step, self.max_delay + 1, self.step))

        def get_delays(start, end):
            res = []
            diff = end - start
            if diff < step:
                return res
            i = -1
            while True:
                shift_i = shifts[i]
                diff_new = diff - shift_i
                if diff_new < 0:
                    i -= 1
                elif diff_new < step:
                    res.append(shift_i)
                    return res
                else:
                    res.append(shift_i)
                    diff = diff_new

        sorted_items = sorted(bin2events.items(), key=lambda x: x[0])
        tokens = []
        for i in range(len(sorted_items) - 1):
            tokens += sorted_items[i][1]
            start = sorted_items[i][0] * self.step
            end = sorted_items[i + 1][0] * self.step
            delays = get_delays(start, end)
            tokens += sorted([f"TIME_SHIFT_{x}" for x in delays], key=lambda x: int(x.split("_")[-1]), reverse=True)
        return tokens

    def tokens2midi(self, tokens):
        """
        Игнорирование токенов педали
        :param tokens:
        :return:
        """
        midi = PrettyMIDI()
        program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = Instrument(program=program)
        velocity = 0
        t = 0
        pitch2start = {}
        pitch2end = {}
        pitch2velocity = {}
        for token in tokens:
            if token.startswith("PEDAL"):
                continue
            value = int(token.split("_")[-1])
            if token.startswith("SET_VELOCITY"):
                velocity = value
            if token.startswith("TIME_SHIFT"):
                t += value
                pitch2end = {k: v + value for k, v in pitch2end.items()}
            if token.startswith("NOTE_ON"):
                pitch2start[value] = t
                pitch2end[value] = t
                pitch2velocity[value] = velocity
            if token.startswith("NOTE_OFF"):
                if value in pitch2start:
                    start = pitch2start.pop(value)
                    end = pitch2end.pop(value)
                    if end > start:
                        note = Note(
                            velocity=self._bin2velocity(pitch2velocity.pop(value)),
                            pitch=value,
                            start=start / 1000,
                            end=end / 1000
                        )
                        piano.notes.append(note)
        midi.instruments.append(piano)
        return midi


def generate_predictions(
        init_tokens: List[str],
        gen,
        converter: Union[DataConverterV1, DataConverterV2],
        midi_dir: str,
        num_steps: int = 512,
        max_sequence_for_step: int = None,
):
    """
    Генерация предиктов
    :param init_tokens: токена начала последовательности
    :param gen: maestro.MelodyGenerator - вспомогательный класс для генерации последовательности с помощью модели
    :param converter: вспомогательный класс для конвертации токенов в MIDI
    :param midi_dir: папка сохранения результатов
    :param num_steps: число токенов для генерации
    :param max_sequence_for_step: максимальное число токенов, которое тебуется учитывать на каждом шаге генерации
    :return:
    """
    os.makedirs(midi_dir, exist_ok=True)
    tokens_sample = gen.decode(init_tokens, num_steps, max_sequence_for_step,)
    midi_init = converter.tokens2midi(init_tokens)
    midi_sample = converter.tokens2midi(tokens_sample)
    files = os.listdir(midi_dir)
    if files:
        i = max(int(x.split("_")[0]) for x in files) + 1
    else:
        i = 0
    midi_init.write(os.path.join(midi_dir, f"{i}_init.mid"))
    midi_sample.write(os.path.join(midi_dir, f"{i}_sample.mid"))


def load_data(data_dir: str, converter: Union[DataConverterV1, DataConverterV2]):
    """
    Перевод MIDI-файлов в последовательности токенов событий
    :param data_dir: распакованный архив
    :param converter: вспомогательный класс для ковертации MIDI-файлов в токены событий
    :param step: минимальный временной интервал между событиями (в мс)
    :param max_delay: максимальный временной интервал между событиями (в мс), соответствующий одному токену
    :return:
    """
    with open(os.path.join(data_dir, "maestro-v2.0.0.json")) as f:
        d = json.load(f)

    file2split = {x["midi_filename"]: x["split"] for x in d}

    train_tokens = []
    eval_tokens = []
    test_tokens = []

    for year in os.listdir(data_dir):
        if year.isdigit():
            year_dir = os.path.join(data_dir, year)
            print(year_dir)
            for file in tqdm(os.listdir(year_dir)):
                midi_file_name = os.path.join(year_dir, file)
                midi = pretty_midi.PrettyMIDI(midi_file_name)
                tokens = converter.midi2tokens(midi)
                split = file2split[year + "/" + file]
                if split == "train":
                    train_tokens.append(tokens)
                elif split == "validation":
                    eval_tokens.append(tokens)
                elif split == "test":
                    test_tokens.append(tokens)
                else:
                    raise
    return train_tokens, eval_tokens, test_tokens


def get_model_data(data: List[List[str]], num_samples: int, seq_len: int, token2id: dict, step: int, max_delay: int,
                   augment: bool, clean_sample: bool = False) -> List[List[int]]:
    """
    Создание обучающей выборки (выборки для валидации)
    :param data: один из элементов результата функции load_data
    :param num_samples: число объектов обучения
    :param seq_len: длина обучающего примера
    :param token2id: отобржение токен события -> int
    :param step: см. load_data
    :param max_delay: см. load_data
    :param augment: нужна ли аугментация
    :param clean_sample: удалять ли токены NOTE_OFF_*, PEDAL_OFF, для которых нет соответствующих токенов начала
    :return: обучающая выборка (выборка для валидации)
    """
    sequences = []
    seq_len_plus_one = seq_len + 1

    tempo_coefs = [0.95, 0.975, 1.0, 1.025, 1.05]
    pitch_shifts = [-3, -2, -1, 0, -1, 2, 3]

    def aug(tokens):
        shift = random.choice(pitch_shifts)
        coef = random.choice(tempo_coefs)
        tokens_encoded = []
        for t in tokens:
            if t.startswith("NOTE"):
                p, q, v = t.split("_")
                v_new = int(v) + shift
                t_new = "_".join((p, q, str(v_new)))
            elif t.startswith("TIME"):
                p, q, v = t.split("_")
                m = max_delay // step
                v_new = step * min(m, round(int(v) * coef / step))
                t_new = "_".join((p, q, str(v_new)))
            else:
                t_new = t
            tokens_encoded.append(token2id[t_new])
        return tokens_encoded

    def sample_sequence(example):
        """
        Из последовательности должны быть удалены токены PEDAL_OFF NOTE_OFF_*, не содержащие
        соответствующих токенов PEDAL_ON, NOTE_ON_*
        Последовательность должна быть длины n
        """
        while True:
            start = random.randint(0, len(example) - seq_len_plus_one)
            end = start + seq_len_plus_one
            candidate = example[start:end]
            started_notes = set()
            started_pedal = True
            subseq = []
            for t in candidate:
                if t.startswith("NOTE"):
                    pitch = t.split("_")[-1]
                    if t.startswith("NOTE_ON"):
                        started_notes.add(pitch)
                    elif t.startswith("NOTE_OFF"):
                        if pitch not in started_notes:
                            continue
                if t == "PEDAL_ON":
                    started_pedal = True
                if t == "PEDAL_OFF":
                    if not started_pedal:
                        continue
                subseq.append(t)
            diff = seq_len_plus_one - len(subseq)
            subseq += example[end:end + diff]
            if len(subseq) == seq_len_plus_one:
                return subseq

    pbar = tqdm(total=num_samples)
    while len(sequences) < num_samples:
        example = random.choice(data)
        if len(example) >= seq_len_plus_one:
            if clean_sample:
                subseq = sample_sequence(example)
            else:
                start = random.randint(0, len(example) - seq_len_plus_one)
                end = start + seq_len_plus_one
                subseq = example[start:end]
            if augment:
                subseq = aug(subseq)
            else:
                subseq = [token2id[t] for t in subseq]
            sequences.append(subseq)
            pbar.update()
    pbar.close()
    return sequences
