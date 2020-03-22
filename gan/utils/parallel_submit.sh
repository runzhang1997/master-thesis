for file in ../../sentinel_data/test_data/unzip/*; do
    echo "$file"
    bsub < generate_training_data.sh $file
done
  