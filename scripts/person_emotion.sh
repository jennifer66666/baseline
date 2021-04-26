# arrange the original train dir
# from train/persons/emotions/angles
# to ds_train/person_emotions/abgles

# to run the script we stand on the root baseline/

s="try"
t="try_arrange"

for i in dataset/"$s"/*
do
    person=$(basename "$i")
    #echo $person
    for j in dataset/"$s"/"$person"/*
    do
        emotion=$(basename "$j")
        if [[ "$emotion" != "info.txt" ]]
        then
            #echo $emotion
            cp -r dataset/"$s"/"$person"/"$emotion" dataset/"$t"/"$person""_""$emotion"
        fi
        
    done
done

