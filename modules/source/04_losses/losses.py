class MSELoss:
    '''Mean Squared Error losses of regression tasks. '''

    def __init__(self):
        '''Initialize the MSE loss Function'''
        pass

    def forward(self,predictions: Tensor, targets:Tensor):
        '''
        compute the Mean Squared Error Loss between predictions and targets

        TODO: Implement the MSELoss calculation

        APPROACH:
        1. Compute difference predictions-targets
        2. Square the differences :diff2
        3. Take mean across all elements

        EXAMPLE:
        >>> loss_fn = MSELoss()
        >>> predictions = Tensor([1,2,3])
        >>> targets = Tensor([3,4,5])
        >>> loss = loss_fn(predictions,targets)
        >>> print(f'MSEloss : {loss.data:.4f}')

          '''
class CrossEntropy:
    """Cross Entropy for classification"""
    def __init__(self):
       """Initialize the CrossEntropy loss Fuction"""
       pass

    def forward(self,logits: Tensor,targets: Tensor):
        """
        compute the cross-entropy loss with numerical stability

        TODO: Implement the CrossEntropy calculation

        APPROACH:
        1.Compute the log-softmax of logits(numerically stable)
        2.Select log-probability for correct classes
        3.Return negative mean of selected probabilities

        EXAMPLE:
        >>> loss_fn = CrossEntropy()
        >>> predictions = Tensor([[1,2,3],[2,5,1]])
        >>> label = Tensor([1,0])
        >>> loss = loss_fn(predictions,label)
        >>> print(f'CrossEntropy loss :{loss.data:.4f}')
        """





