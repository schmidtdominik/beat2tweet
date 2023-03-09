import csv
import itertools

from muscaps.caption_evaluation_tools.eval_metrics import evaluate_metrics

preds = [
        "<sos> this is a classical music piece there is a <unk> of <unk> <unk> and <unk> there is a <unk> and <unk> and",
        "<sos> this song contains a male voice singing in a higher register along with a piano playing chords in the background the song",
]

refs = [
        [
            "<sos> this song is a classical composition played on a harpsichord this features the notes played on the higher register and the bass <eos>"
        ],
        [
            "<sos> the low quality recording features a traditional pop song that consists of harmonized vocals singing over breathy flute melody groovy bass wooden <eos>"
        ]]
       

refs = list(itertools.chain(*refs))

header = ['file_name', 'caption_predicted']
with open('preds.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(len(preds)):
      # write the data
      writer.writerow([i,preds[i]])

header = ['file_name', 'caption_reference_01']
with open('refs.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(len(refs)):
      # write the data
      writer.writerow([i,refs[i]])

evaluate_metrics('preds.csv', 'refs.csv',nb_reference_captions=1)