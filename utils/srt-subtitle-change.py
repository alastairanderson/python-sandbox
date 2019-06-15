import math

'''
Changes the time in .srt subtitle films by a specified offset (OFFSET_IN_SECONDS)
'''


FILE_NAME = 'S04E09.srt'
OFFSET_IN_SECONDS = -66
SECONDS_IN_MINUTE = 60

# 1. Open file
text_file = open(FILE_NAME, "r")
subtitles = text_file.readlines()
result = []

# 2. Read it all line by line
for line in subtitles:
    updated_line = line.strip()

    if '00:' in updated_line[0:3]:
        times = updated_line.split(' --> ')

        start_seconds = (int(times[0][3:5]) * SECONDS_IN_MINUTE) + int(times[0][6:8]) + OFFSET_IN_SECONDS
        end_seconds = (int(times[1][3:5]) * SECONDS_IN_MINUTE) + int(times[1][6:8]) + OFFSET_IN_SECONDS
        
        s_start = start_seconds % SECONDS_IN_MINUTE
        m_start = math.floor(start_seconds / SECONDS_IN_MINUTE)
        ms_start = times[0][9:12]

        s_end = end_seconds % SECONDS_IN_MINUTE
        m_end = math.floor(end_seconds / SECONDS_IN_MINUTE)
        ms_end = times[1][9:12]

        new_start = "00:{0}:{1},{2}".format(f'{m_start:02}', f'{s_start:02}', ms_start)
        new_end = "00:{0}:{1},{2}".format(f'{m_end:02}', f'{s_end:02}', ms_end)

        new_line = "{0} --> {1}".format(new_start, new_end)

        result.append(new_line)
    else:
        result.append(updated_line)


with open(FILE_NAME, 'w') as f:
    for item in result:
        f.write("%s\n" % item)