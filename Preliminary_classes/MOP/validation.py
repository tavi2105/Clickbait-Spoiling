all_keys = ['targetDescription', 'postPlatform', 'targetTitle', 'spoilerPositions', 'targetParagraphs', 'spoiler',
            'targetMedia', 'uuid', 'postId', 'tags', 'provenance', 'targetKeywords', 'postText', 'targetUrl']
important_keys = ["uuid", "postText", "targetParagraphs", "targetTitle", "targetUrl"]
good_tags = ["passage", "phrase"]


def clean_data(data):
    print(">>>>>>>>>>>>> /|\\  Cleaning data  /|\\ <<<<<<<<<<<<<")
    for i, one_entry in enumerate(data):
        if set(all_keys) != set(one_entry.keys()):
            data.pop(i)
            continue

        for key in important_keys:
            if not data[i][key]:
                data.pop(i)
                continue

        for tag in data[i]["tags"]:
            if tag not in good_tags:
                data.pop(i)
                continue

    return data


def validate_data(data):
    print(">>>>>>>>>>>>> \\|/ Validating data \\|/ <<<<<<<<<<<<<")

    for one_entry in data:
        if set(all_keys) != set(one_entry.keys()):
            return False

        for key in important_keys:
            if not one_entry[key]:
                return False

        for tag in one_entry["tags"]:
            if tag not in good_tags:
                return False

    return True
