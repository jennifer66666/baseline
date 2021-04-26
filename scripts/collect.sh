# collect person_emotion with stack_landmarks computed

# if validation, comment the "label" line
# and if [[ "$num_file" -eq "6" ]] -> if [[ "$num_file" -eq "4" ]]
s="validation_arrange"
t="validation_FAN"

for i in dataset/"$s"/*
do
    person_emotion=$(basename "$i")
    #echo $person_emotion
    num_file=`ls dataset/"$s"/"$person_emotion"/"Images" | wc -l`
    # 0 1 2 fusion label_landmarks stack_landmarks
    if [[ "$num_file" -eq "6" ]]
    then
        mkdir dataset/"$t"/"$person_emotion"
        cp dataset/"$s"/"$person_emotion"/"Images/label_landmarks.csv" dataset/"$t"/"$person_emotion"/"label_landmarks.csv"
        cp dataset/"$s"/"$person_emotion"/"Images/stack_landmarks.csv" dataset/"$t"/"$person_emotion"/"stack_landmarks.csv"
    fi   
done