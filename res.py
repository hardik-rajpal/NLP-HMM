import re
isAmount = lambda word:(re.compile('\W\d+').match(word)!=None)
