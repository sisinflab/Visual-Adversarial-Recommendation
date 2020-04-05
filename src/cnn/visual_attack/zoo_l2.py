from cnn.visual_attack.l2_attack_black import *
from skimage.transform import rescale

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')


class ZOOL2(BlackBoxL2):
    def __init__(self, sess, model, height=224, width=224, batch_size=64, num_channels=3, num_labels=1000,
                 confidence=CONFIDENCE, targeted=False, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=20000, print_every=5,
                 early_stop_iters=0, abort_early=ABORT_EARLY, initial_const=10.0, use_log=False, use_tanh=False,
                 use_resize=False, adam_beta1=0.9, adam_beta2=0.999, reset_adam_after_found=True, solver="adam",
                 save_ckpts="", load_checkpoint="", start_iter=0, init_size=32, use_importance=True):
        """
        The L_2 optimized attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of gradient evaluations to run simultaneously.
        targeted: True if we should perform a targeted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal trade-off-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial trade-off-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """

        super(ZOOL2, self).__init__(sess, model, height, width, batch_size, num_channels, num_labels,
                                    confidence, targeted, learning_rate, binary_search_steps,
                                    max_iterations, print_every, early_stop_iters, abort_early, initial_const, use_log,
                                    use_tanh, use_resize, adam_beta1, adam_beta2, reset_adam_after_found, solver,
                                    save_ckpts, load_checkpoint, start_iter, init_size, use_importance)

    def get_new_prob(self, prev_modifier, gen_double=False):
        prev_modifier = np.squeeze(prev_modifier)
        old_shape = prev_modifier.shape
        if gen_double:
            new_shape = (old_shape[0] * 2, old_shape[1] * 2, old_shape[2])
        else:
            new_shape = old_shape
        prob = np.empty(shape=new_shape, dtype=np.float32)
        for i in range(prev_modifier.shape[2]):
            image = np.abs(prev_modifier[:, :, i])
            image_pool = self.max_pooling(image, old_shape[0] // 8)
            if gen_double:
                # substituted scipy.misc.imresize with skimage.transform.rescale
                prob[:, :, i] = rescale(image=image_pool, scale=2.0, order=0)
            else:
                prob[:, :, i] = image_pool
        prob /= np.sum(prob)
        return prob
