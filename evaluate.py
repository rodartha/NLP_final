import os
import json


def main():
    #reformat_hans()
    #hans_misses()
    #find_misses()
    #hypothesis_only()
    #percent_misses_each_category()
    check_matches_hans_no_hans()

    print("Nothing to run at the moment")

def check_matches_hans_no_hans():
    filename1 = 'eval_output/incorrect_predictions.jsonl'

    with open(filename1, mode='r') as f:
        og_eval_misses = [json.loads(line) for line in f][0]

    filename2 = 'eval_output_hans_smaller/incorrect_predictions.jsonl'

    with open(filename2, mode='r') as f:
        hans_eval_misses = [json.loads(line) for line in f][0]

    joint_misses = []
    for og_line in og_eval_misses:
        for hans_line in hans_eval_misses:
            if og_line['premise'] == hans_line['premise'] and og_line['hypothesis'] == hans_line['hypothesis']:
                joint_misses.append(og_line)
                break

    print(len(joint_misses))

    outfile = 'comparisons/hans_smaller_vs_og_on_snli.jsonl'

    with open(outfile, mode='w') as f:
        json.dump(joint_misses, f)


def percent_misses_each_category():
    filename = 'eval_output_hans_smaller/incorrect_predictions.jsonl'

    with open(filename, mode='r') as f:
        evaluation_data = [json.loads(line) for line in f][0]

    amount_each_predicted_label = {
        'entailment': {
            'total': 0,
            'percent': 0,
            'real_label': {
                'neutral': 0,
                'non-entailment': 0
            }
        },
        'neutral': {
            'total': 0,
            'percent': 0,
            'real_label': {
                'entailment': 0,
                'non-entailment': 0
            }
        },
        'non-entailment': {
            'total': 0,
            'percent': 0,
            'real_label': {
                'neutral': 0,
                'entailment': 0
            }
        },
        'total_incorrect': len(evaluation_data)
    }

    for line in evaluation_data:
        if line['predicted_label'] == 0:
            amount_each_predicted_label['entailment']['total'] += 1
            if line['label'] == 1:
                amount_each_predicted_label['entailment']['real_label']['neutral'] += 1
            else:
                amount_each_predicted_label['entailment']['real_label']['non-entailment'] += 1
        elif line['predicted_label'] == 1:
            amount_each_predicted_label['neutral']['total'] += 1
            if line['label'] == 0:
                amount_each_predicted_label['neutral']['real_label']['entailment'] += 1
            else:
                amount_each_predicted_label['neutral']['real_label']['non-entailment'] += 1
        else:
            amount_each_predicted_label['non-entailment']['total'] += 1
            if line['label'] == 0:
                amount_each_predicted_label['non-entailment']['real_label']['entailment'] += 1
            else:
                amount_each_predicted_label['non-entailment']['real_label']['neutral'] += 1

    for key in amount_each_predicted_label.keys():
        if key == 'total_incorrect':
            pass
        else:
            amount_each_predicted_label[key]['percent'] = amount_each_predicted_label[key]['total'] / amount_each_predicted_label['total_incorrect']

    print(amount_each_predicted_label)


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

def find_misses():
    filename = 'eval_output_hans_smaller/eval_predictions.jsonl'

    with open(filename, mode='r') as f:
        evaluation_data = [json.loads(line) for line in f]

    incorrect_data = []

    for line in evaluation_data:
        if line['label'] == line['predicted_label']:
            pass
        else:
            incorrect_data.append(line)

    outfile = 'eval_output_hans_smaller/incorrect_predictions.jsonl'

    with open(outfile, mode='w') as f:
        json.dump(incorrect_data, f)

def hans_misses():
    filename = 'eval_output_hans_trained_on_hans/eval_predictions.jsonl'

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

    outfile = 'eval_output_hans_trained_on_hans/incorrect_predictions.jsonl'

    with open(outfile, mode='w') as f:
        json.dump(incorrect_data, f)


def reformat_hans():
    filename = 'hans_training/heuristics_train_set.jsonl'

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

    outfile = 'hans_training/hans_train_reformatted.jsonl'

    print(possible_labels)

    with open(outfile, mode='w') as f:
        json.dump(formatted_data, f)


if __name__ == "__main__":
    main()
