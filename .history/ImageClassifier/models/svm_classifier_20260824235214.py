import torch

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


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, W, X_batch, y_batch, reg):
    return svm_loss_vectorized(W, X_batch, y_batch, reg)

