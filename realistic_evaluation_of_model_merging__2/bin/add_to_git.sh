files=("pretrained_mt5-xl-lm-adapt.pt" "models/squad/checkpoint.pt" "models/tydiqa/fisher.pt" "models/xlwic/covariance.pt" "models/squad/covariance.pt" "models/tydiqa/trimmed.pt" "models/xlwic/fisher.pt" "models/squad/fisher.pt" "models/wiki_lingua/checkpoint.pt" "models/xlwic/trimmed.pt" "models/squad/trimmed.pt" "models/wiki_lingua/covariance.pt" "models/xnli/covariance.pt" "models/tydiqa/checkpoint.pt" "models/wiki_lingua/fisher.pt" "models/xnli/fisher.pt" "models/tydiqa/covariance.pt" "models/wiki_lingua/trimmed.pt" "models/xnli/trimmed.pt"  "models/xnli/checkpoint.pt" "models/xlwic/checkpoint.pt")
for file in "${files[@]}"
do
        git theta track "$file"
        GIT_THETA_MAX_CONCURRENCY=50 git add "$file"
        git commit -m "$file"
done