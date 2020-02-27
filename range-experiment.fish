#!/usr/bin/fish

set ranges 10 11 12 13 14

for r in $ranges
  echo "range = $r"
  set range_condition "s/\"range\": .*,\$/\"range\": $r,/g"
  sed -ie (echo $range_condition) config.json
  set r_condition "s/\"r\": .*,\$/\"r\": $r,/g"
  sed -ie (echo $r_condition) config.json
  set save_path_condition "s/\"save_path\": \(.*\)range.*-sample.csv/\"save_path\": \1range$r-sample.csv/g"
  sed -ie (echo $save_path_condition) config.json
  ./lsh
end
