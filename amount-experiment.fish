#!/usr/bin/fish

set amounts 1000 1500 2000 2500 3000

for a in $amounts
  echo "amount = $a"
  set amount_condition "s/\"n\": .*,\$/\"n\": $a,/g"
  sed -ie (echo $amount_condition) config.json
  set save_path_condition "s/\"save_path\": \(.*\)data.*k/\"save_path\": \1data$a""k/g"
  sed -ie (echo $save_path_condition) config.json
  ./lsh
end
