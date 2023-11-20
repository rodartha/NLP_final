import os
import json


def main():
    #filename = 'eval_output/eval_predictions.jsonl'
    """
    filename = 'analysis_sets/heuristics_evaluation_set.jsonl'

    with open(filename, mode='r') as f:
        evaluation_data = [json.loads(line) for line in f]

    print(len(evaluation_data))
    print(evaluation_data[0].keys())

    for key in evaluation_data[0].keys():
        print(key)
        print(evaluation_data[10000][key])
    """

    #reformat_hans()
    hans_misses()
    #hypothesis_only()

def hypothesis_only():
    filename = 'eval_output/eval_predictions.jsonl'

    with open(filename, mode='r') as f:
        evaluation_data = [json.loads(line) for line in f]

    hypothesis_only_data = []
    for line in evaluation_data:
        line['premise'] = ''
        hypothesis_only_data.append(line)

    outfile = 'analysis_sets/eval_hypo_only.jsonl'

    with open(outfile, mode='w') as f:
        json.dump(hypothesis_only_data, f)

def hans_misses():
    filename = 'hans_analysis/eval_predictions.jsonl'

    with open(filename, mode='r') as f:
        evaluation_data = [json.loads(line) for line in f]

    incorrect_data = []
    class_of_error = {}
    class_correct = {}
    for line in evaluation_data:
        if line['label'] == line['predicted_label']:
            if line['heuristic'] in class_correct.keys():
                class_correct[line['heuristic']] += 1
            else:
                class_correct[line['heuristic']] = 1
        else:
            incorrect_data.append(line)
            if line['heuristic'] in class_of_error.keys():
                class_of_error[line['heuristic']] += 1
            else:
                class_of_error[line['heuristic']] = 1

    print("CLASS OF ERRORS:")
    print(class_of_error)
    print("CLASS OF CORRECT:")
    print(class_correct)

    outfile = 'hans_analysis/incorrect_predictions.jsonl'

    with open(outfile, mode='w') as f:
        json.dump(incorrect_data, f)


def reformat_hans():
    filename = 'analysis_sets/heuristics_evaluation_set.jsonl'

    with open(filename, mode='r') as f:
        evaluation_data = [json.loads(line) for line in f]

    formatted_data = []
    possible_labels = set()
    for line in evaluation_data:
        new_data_object = {}
        for key in line.keys():
            if key == 'sentence1':
                new_data_object['premise'] = line[key]
            elif key == 'sentence2':
                new_data_object['hypothesis'] = line[key]
            elif key == 'gold_label':
                if line[key] == 'entailment':
                    new_data_object['label'] = 0
                else:
                    new_data_object['label'] = 2
                possible_labels.add(line[key])
            else:
                new_data_object[key] = line[key]
        formatted_data.append(new_data_object)

    outfile = 'analysis_sets/hans_reformatted.jsonl'

    print(possible_labels)

    with open(outfile, mode='w') as f:
        json.dump(formatted_data, f)


if __name__ == "__main__":
    main()
