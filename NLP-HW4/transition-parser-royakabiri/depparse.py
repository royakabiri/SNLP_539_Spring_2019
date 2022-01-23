from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Sequence, Text, Union
import re
from collections import deque
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

@dataclass()
class Dep:
    """A word in a dependency tree.

    The fields are defined by https://universaldependencies.org/format.html.
    """
    id: Text
    form: Union[Text, None]
    lemma: Union[Text, None]
    upos: Text
    xpos: Union[Text, None]
    feats: Sequence[Text]
    head: Union[Text, None]
    deprel: Union[Text, None]
    deps: Sequence[Text]
    misc: Union[Text, None]

def read_conllu(path: Text) -> Iterator[Sequence[Dep]]:
    """Reads a CoNLL-U format file into sequences of Dep objects.

    The CoNLL-U format is described in detail here:
    https://universaldependencies.org/format.html
    A few key highlights:
    * Word lines contain 10 fields separated by tab characters.
    * Blank lines mark sentence boundaries.
    * Comment lines start with hash (#).

    Each word line will be converted into a Dep object, and the words in a
    sentence will be collected into a sequence (e.g., list).

    :return: An iterator over sentences, where each sentence is a sequence of
    words, and each word is represented by a Dep object.
    """
    #a function to return an empty list for underscore,
    #in the 5 and 8 indices which are themselves a sequence
    def turn_to_emptyList(text):
        if text=='_':
            return []
        else:
            #if there is not an underscore, so split by |
            return text.split("|")

    #a function to return None where there is an underscore in any other positions,
    def turn_to_None(text):
        if text == '_':
            return None
        else:
            return text  

    
    #read the text file line by line  
    with open(path,"r") as file:
        #a list to store the sequence of words with their Dep objects
        sentSequence=[]
        #for each line
        for line in file:
            #if the line is a comment, just continue
            if line.startswith("#"):
                continue
            #if it is not an empty line 
            elif line.strip() != '':
                #split the line by tab, 
                id, form, lemma, upos, xpos,feats, head, deprel, deps, misc = line.strip().split("\t")

                #create the obj sequence from the splitted fields
                #change the underscores according to the above functions 
                objectSequence=Dep(id,form,lemma,upos,\
                turn_to_None(xpos), turn_to_emptyList(feats), turn_to_None(head),\
                turn_to_None(deprel), turn_to_emptyList(deps),turn_to_None(misc))
                #append the Dep object to the sequence list
                sentSequence.append(objectSequence)
            #if there is an empty line, it means that it's the end of the sentence
            else:
                #so return the whole sentence with the dep objects for each word
                # print(sentSequence)
                yield sentSequence
                #and then create an empty list to store the next sentence 
                sentSequence=[]

class Action(Enum):
    """An action in an "arc standard" transition-based parser."""
    SHIFT = 1
    LEFT_ARC = 2
    RIGHT_ARC = 3


def parse(deps: Sequence[Dep],
          get_action: Callable[[Sequence[Dep], Sequence[Dep]], Action]) -> None:
    """Parse the sentence based on "arc standard" transitions.

    Following the "arc standard" approach to transition-based parsing, this
    method creates a stack and a queue, where the input Deps start out on the
    queue, are moved to the stack by SHIFT actions, and are combined in
    head-dependent relations by LEFT_ARC and RIGHT_ARC actions.

    This method does not determine which actions to take; those are provided by
    the `get_action` argument to the method. That method will be called whenever
    the parser needs a new action, and then the parser will perform whatever
    action is returned. If `get_action` returns an invalid action (e.g., a
    SHIFT when the queue is empty), an arbitrary valid action will be taken
    instead.

    This method does not return anything; it modifies the `.head` field of the
    Dep objects that were passed as input. Each Dep object's `.head` field is
    assigned the value of its head's `.id` field, or "0" if the Dep object is
    the root.

    :param deps: The sentence, a sequence of Dep objects, each representing one
    of the words in the sentence.
    :param get_action: a function or other callable that takes the parser's
    current stack and queue as input, and returns an "arc standard" action.
    :return: Nothing; the `.head` fields of the input Dep objects are modified.
    """

    stack=[]
    queue=deque(deps)
    while len(queue)!= 0 or len(stack)>1:
        action=get_action(stack, queue)
        
        if action == Action.SHIFT and len(queue) == 0:
        	stack.pop()

        if len(stack)<2 and action==Action.LEFT_ARC:
            action=Action.SHIFT

        if len(stack)<2 and action==Action.RIGHT_ARC:
            action=Action.SHIFT

        if action == Action.SHIFT and len(queue)!= 0:
        	word=queue.popleft()
        	stack.append(word)

        elif action == Action.LEFT_ARC:
        	topStack=stack.pop()
        	belowStack=stack.pop()
        	belowStack.head=topStack.id
        	stack.append(topStack)

        elif action == Action.RIGHT_ARC:
        	topStack=stack.pop()
        	belowStack=stack.pop()
        	topStack.head= belowStack.id
        	stack.append(belowStack) 

    #set the head value of the root to zero 
    stack[0].head="0" 

#a function to extract the feature in the oracle part 
def feature_extraction(stack, queue):
        	
    if len(queue)>0 and len(stack)>1:
        top_stack=stack[-1]
        top_stack2=stack[-2]
        top_queue=queue[0]
        featureValueMap={"token.stack":top_stack.form ,"pos.stack":top_stack.upos, 
        "token.queue":top_queue.form ,"pos.queue":top_queue.upos,
        "token.stack2":top_stack2.form, "pos.stack2":top_stack2.upos}
        

    elif len(queue)==0 and len(stack)>1:
        top_stack=stack[-1]
        top_stack2=stack[-2]
        featureValueMap={"token.stack":top_stack.form ,"pos.stack":top_stack.upos,
        "token.stack2":top_stack2.form, "pos.stack2":top_stack2.upos}
	   

    elif len(stack)==0 and len(queue)>0:
        top_queue=queue[0]
        featureValueMap={"token.queue":top_queue.form ,"pos.queue":top_queue.upos}

    else:  #len of queue is 0
        featureValueMap={"end-of-parse":'random value'}
        

    return featureValueMap
      		

class Oracle:
    def __init__(self, deps: Sequence[Dep]):
        """Initializes an Oracle to be used for the given sentence.

        Minimally, it initializes a member variable `actions`, a list that
        will be updated every time `__call__` is called and a new action is
        generated.

        Note: a new Oracle object should be created for each sentence; an
        Oracle object should not be re-used for multiple sentences.

        :param deps: The sentence, a sequence of Dep objects, each representing
        one of the words in the sentence.
        """
    
        #a list to  store all the actions in the --call-- method
        self.actions=[]
        #a list to store all the features 
        self.features=[]

        #a dict to store all the info of the dependents of each head
        self.depedents_tracking_Dict=defaultdict(list)   
        for dep in deps:
        	current_dep_id=dep.id
        	current_dep_head_Number=dep.head
        	#store the dependet id as the value of the key head in the dict
        	self.depedents_tracking_Dict[current_dep_head_Number].append(current_dep_id)


    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]) -> Action:
        """Returns the Oracle action for the given "arc standard" parser state.

        The oracle for an "arc standard" transition-based parser inspects the
        parser state and the reference parse (represented by the `.head` fields
        of the Dep objects) and:
        * Chooses LEFT_ARC if it produces a correct head-dependent relation
          given the reference parse and the current configuration.
        * Otherwise, chooses RIGHT_ARC if it produces a correct head-dependent
          relation given the reference parse and all of the dependents of the
          word at the top of the stack have already been assigned.
        * Otherwise, chooses SHIFT.

        The chosen action should be both:
        * Added to the `actions` member variable
        * Returned as the result of this method

        Note: this method should only be called on parser state based on the Dep
        objects that were passed to __init__; it should not be used for any
        other Dep objects.

        :param stack: The stack of the "arc standard" transition-based parser.
        :param queue: The queue of the "arc standard" transition-based parser.
        :return: The action that should be taken given the reference parse
        (the `.head` fields of the Dep objects).
        """
        while len(queue)!= 0 or len(stack)>1:
        	#extract the features
        	feature=feature_extraction(stack, queue)
        	#store them in the feature list 
        	self.features.append(feature)

        	if len(stack)>= 2:
        		above_stack=stack[-1]
        		below_stack=stack[-2]
        		if below_stack.head==above_stack.id and len(self.depedents_tracking_Dict[below_stack.id])==0:
        			action=Action.LEFT_ARC
        			self.actions.append(action)
        			#delete the dependent id from the list of dependents of that head
        			self.depedents_tracking_Dict[above_stack.id].remove(below_stack.id)
                #check if the dependent list of the child node is empty
        		elif above_stack.head == below_stack.id and len(self.depedents_tracking_Dict[above_stack.id])==0:
        			action=Action.RIGHT_ARC
        			self.actions.append(action)
        			#remove this child node from the dependent list of the head
        			self.depedents_tracking_Dict[below_stack.id].remove(above_stack.id) 
        		#otherwise, take the Shift action
       			else:
       				action=Action.SHIFT
       				self.actions.append(action)
       		else:
       			action=Action.SHIFT
       			self.actions.append(action)

       		
       		return action 


class Classifier:
    def __init__(self, parses: Iterator[Sequence[Dep]]):
        """Trains a classifier on the given parses.

        There are no restrictions on what kind of classifier may be trained,
        but a typical approach would be to
        1. Define features based on the stack and queue of an "arc standard"
           transition-based parser (e.g., part-of-speech tags of the top words
           in the stack and queue).
        2. Apply `Oracle` and `parse` to each parse in the input to generate
           training examples of parser states and oracle actions. It may be
           helpful to modify `Oracle` to call the feature extraction function
           defined in 1, and store the features alongside the actions list that
           `Oracle` is already creating.
        3. Train a machine learning model (e.g., logistic regression) on the
           resulting features and labels (actions).

        :param parses: An iterator over sentences, where each sentence is a
        sequence of words, and each word is represented by a Dep object.
        """
        self.vectorizer = DictVectorizer() 
        self.encoder=preprocessing.LabelEncoder()
        self.logisticRegr = LogisticRegression(penalty='l2',solver='sag', multi_class="auto", C=5, max_iter=1000)
        
        feature_list=[]
        action_list=[]

        #for each sentence in the input sentences 
        for sent in parses:
            oracle= Oracle(sent)
            sent_Parse=parse(sent,oracle)  
            feature_list.extend(oracle.features)  
            action_list.extend(oracle.actions) 

        self.vectorizer.fit(feature_list)
        #transfer action_list to string or numbers before you fit it into the label encoder
        #since label encoder cannot take an object as an input
        new_action_list=[]
        for action in action_list: 
        	#just take the value 
        	new_action_list.append(action.value)

        self.encoder.fit(new_action_list)

        #train the model using the features and actions
        self.logisticRegr.fit(self.vectorizer.transform (feature_list), self.encoder.transform(new_action_list)) 

    def __call__(self, stack: Sequence[Dep], queue: Sequence[Dep]) -> Action:
        """Predicts an action for the given "arc standard" parser state.

        There are no restrictions on how this prediction may be made, but a
        typical approach would be to convert the parser state into features,
        and then use the machine learning model (trained in `__init__`) to make
        the prediction.

        :param stack: The stack of the "arc standard" transition-based parser.
        :param queue: The queue of the "arc standard" transition-based parser.
        :return: The action that should be taken.
        """
        features=feature_extraction(stack,queue)
        featureMat=self.vectorizer.transform(features) 
        # predict the action for this parser state, given the feature vector
        action=self.logisticRegr.predict(featureMat)
        final_action=self.encoder.inverse_transform(action)
        real_action=Action(final_action)

        return real_action


