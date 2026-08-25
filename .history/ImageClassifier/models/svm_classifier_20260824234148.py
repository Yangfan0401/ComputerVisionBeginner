
def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation. When you implment
  the regularization over W, please DO NOT multiply the regularization term by
  1/2 (no coefficient).

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0 #total_loss
  dW = torch.zeros_like(W) # initialize the gradient as zero

  #############################################################################
  # TODO:  3                                                                   #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  num_train = X.shape[0]
  scores = torch.matmul(W.t(), X.t())
  
  correct_scores = scores[y, torch.arange(num_train)]
  margins = scores.clone()
  margins -= (correct_scores - 1)
  margins[y, torch.arange(num_train)] -= 1
  
  loss = margins[margins>0].sum()
  loss /= num_train
  loss += reg * torch.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:    4                                                                 #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Replace "pass" statement with your code
  # Reuse
  mask_matrix =  (margins > 0).to(torch.float64)
  k = -mask_matrix.sum(dim=0).to(torch.float64)
  mask_matrix[y, torch.arange(num_train)] = k
  dW = torch.matmul(X.t(), mask_matrix.t())
  
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.  When you implment the
  regularization over W, please DO NOT multiply the regularization term by 1/2
  (no coefficient).

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability (Check Numeric Stability #
  # in http://cs231n.github.io/linear-classify/). Don't forget the            #
  # regularization!                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  num_train = X.shape[0]
  all_scores = torch.matmul(W.t(), X.t()) # (C,N)
  max_scores_colmn = all_scores.max(dim=0).values
  correct_scores = torch.exp(all_scores[y, torch.arange(num_train)] - max_scores_colmn)
  sum_scores_colmn = torch.exp(all_scores - max_scores_colmn).sum(dim=0)

  loss = -torch.log(correct_scores / sum_scores_colmn)
  loss = loss.sum()
  loss /= num_train
  loss += reg*torch.sum(W * W)

  all_scores -= max_scores_colmn
  exp_scores = torch.exp(all_scores)
  p_matrix = torch.zeros_like(all_scores)
  p_matrix = exp_scores / exp_scores.sum(dim=0)
  p_matrix[y, torch.arange(num_train)] = (correct_scores / sum_scores_colmn) - 1

  dW = torch.matmul(X.t(), p_matrix.t())
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, W, X_batch, y_batch, reg):
    return svm_loss_vectorized(W, X_batch, y_batch, reg)

