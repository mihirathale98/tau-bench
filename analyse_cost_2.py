import json


def read_file(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_token_info(results):
    return results['token_info']

def main(path):
    results = read_file(path)
    token_info = get_token_info(results)
    print(token_info[0])
if __name__ == '__main__':
    main('res_airline_openai_trial_2/tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4o-llm_0930125611_token_info.json')