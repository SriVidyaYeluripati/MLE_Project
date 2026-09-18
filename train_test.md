# for training the model 
    python main.py play --no-gui --agents my_agent rule_based_agent --train 1 --scenario classic


# train coin heaven 

```
python main.py play \
  --no-gui \
  --agents my_agent \
  --train 1 \
  --scenario coin-heaven \
  --n-rounds 1000
```

# for testing how it is

```
    python main.py play \
  --no-gui \
  --agents my_agent \
  --train 1 \
  --scenario coin-heaven \
  --n-rounds 1000
```

# training in the classic scenario

```
python main.py play \
  --no-gui \
  --agents my_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 3000
```

# then train against easier enemy


```
python main.py play \
  --no-gui \
  --agents my_agent peaceful_agent coin_collector_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 3000
```

# train against rule-based agent
```
python main.py play \
  --no-gui \
  --agents my_agent rule_based_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 5000
```
