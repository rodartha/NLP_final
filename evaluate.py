import os
import json


def main():
    filename = 'eval_output/eval_predictions.jsonl'

    with open(filename, mode='r') as f:
        evaluation_data = [json.loads(line) for line in f]


    print(len(evaluation_data))
    print(evaluation_data[0].keys())


if __name__ == "__main__":
    main()
