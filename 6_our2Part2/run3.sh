
CSV="result.csv"
FIN="finish"


i=0
rm $CSV

while read line
do
    line2=${line%?}
    echo ./main $line2
    ./main $line2
    i=`expr $i + 1`
done < name2.txt

echo $FIN


