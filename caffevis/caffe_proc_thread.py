import time
import cv2
import numpy as np
import ipdb as pdb

from codependent_thread import CodependentThread
from misc import WithTimer
from caffevis_helper import net_preproc_forward



class CaffeProcThread(CodependentThread):
    '''Runs Caffe in separate thread.'''

    def __init__(self, net, upconv_net, state, settings, loop_sleep, pause_after_keys, heartbeat_required, mode_gpu):
        CodependentThread.__init__(self, heartbeat_required)
        self.daemon = True
        self.net = net
        self.settings = settings
        self.upconv_net = upconv_net
        self.image_input = None
        self.upconv_code = None
        self.upconv_code_shape = None
        self.upconv_code_clip = np.load(self.settings.caffevis_upconv_code_clip)
        self.input_dims = self.net.blobs['data'].data.shape[2:4]    # e.g. (227,227)
        self.state = state
        self.last_process_finished_at = None
        self.last_process_elapsed = None
        self.frames_processed_fwd = 0
        self.frames_processed_back = 0
        self.loop_sleep = loop_sleep
        self.pause_after_keys = pause_after_keys
        self.debug_level = 0
        self.mode_gpu = mode_gpu      # Needed so the mode can be set again in the spawned thread, because there is a separate Caffe object per thread.

        # Upconv stuff
        self.upconv_in_layer = 'feat'
        self.upconv_out_layer = 'deconv0'
        # The top left offset that we start cropping the output image to get the 227x227 image
        self.upconv_output_size = self.upconv_net.blobs[self.upconv_out_layer].data.shape[2:4]    # e.g. (227,227)
        self.topleft = ((self.upconv_output_size[0] - self.input_dims[0])/2, (self.upconv_output_size[1] - self.input_dims[1])/2)
        self.image_code_target_layer = None
        self.image_code_target_layer_idx = None
        self.upconv_image_grad_blob = np.zeros((1, 3, self.upconv_output_size[0], self.upconv_output_size[1]), 'float32')
        self.im_upconv_blob = None
        self.im_upconv_crop_blob = None
        self.cached_latest_frame = None
        
    def run(self):
        print 'CaffeProcThread.run called'
        frame = None

        import caffe
        # Set the mode to CPU or GPU. Note: in the latest Caffe
        # versions, there is one Caffe object *per thread*, so the
        # mode must be set per thread! Here we set the mode for the
        # CaffeProcThread thread; it is also set in the main thread.
        if self.mode_gpu:
            caffe.set_mode_gpu()
            print 'CaffeVisApp mode (in CaffeProcThread): GPU'
        else:
            caffe.set_mode_cpu()
            print 'CaffeVisApp mode (in CaffeProcThread): CPU'
        
        while not self.is_timed_out():
            with self.state.lock:
                if self.state.quit:
                    #print 'CaffeProcThread.run: quit is True'
                    #print self.state.quit
                    break
                    
                #print 'CaffeProcThread.run: caffe_net_state is:', self.state.caffe_net_state

                #print 'CaffeProcThread.run loop: next_frame: %s, caffe_net_state: %s, back_enabled: %s' % (
                #    'None' if self.state.next_frame is None else 'Avail',
                #    self.state.caffe_net_state,
                #    self.state.back_enabled)

                frame = None
                run_fwd = False
                run_back = False
                run_upconv = False
                if self.state.caffe_net_state == 'free' and time.time() - self.state.last_key_at > self.pause_after_keys:
                    frame = self.state.next_frame
                    if frame is not None:
                        self.cached_latest_frame = frame
                    self.state.next_frame = None
                    back_enabled = self.state.back_enabled
                    back_mode = self.state.back_mode
                    back_stale = self.state.back_stale
                    #state_layer = self.state.layer
                    #selected_unit = self.state.selected_unit
                    backprop_layer = self.state.backprop_layer
                    backprop_unit = self.state.backprop_unit
                    layer = self.state.layer
                    layer_idx = self.state.layer_idx
                    
                    # Forward should be run for every new frame AND when needed for upconv (e.g. changed selection)
                    run_fwd_normal = frame is not None
                    run_fwd_for_upconv = (self.state.upconv_enabled and
                                          self.cached_latest_frame is not None and
                                          (self.image_code_target_layer is None or self.image_code_target_layer_idx != layer_idx))
                    if run_fwd_for_upconv and frame is None:
                        # Reuse latest frame if needed
                        frame = self.cached_latest_frame
                    run_fwd = run_fwd_normal or run_fwd_for_upconv
                    # Backward should be run if back_enabled and (there was a new frame OR back is stale (new backprop layer/unit selected))
                    run_back = (back_enabled and (run_fwd or back_stale))
                    # Upconv run if it is enabled and an image_layer_code has been stored
                    run_upconv = self.state.upconv_enabled and self.image_code_target_layer is not None
                    self.state.caffe_net_state = 'proc' if (run_fwd or run_back or run_upconv) else 'free'

                    assert not (run_back and run_upconv), 'both should not be enabled at same time (see handle_key)'
            
            #print 'run_fwd,run_back =', run_fwd, run_back
            
            if run_fwd:
                #print 'TIMING:, processing frame'
                self.frames_processed_fwd += 1
                im_small = cv2.resize(frame, self.input_dims)
                with WithTimer('CaffeProcThread:forward', quiet = self.debug_level < 1):
                    net_preproc_forward(self.net, im_small, self.input_dims)
                # Grab layer code for upconv mode
                self.image_input = self.net.blobs['data'].data.copy()
                self.image_code_target_layer = self.net.blobs[layer].data.copy()
                self.image_code_target_layer_idx = layer_idx
                #print 'GRABBED LAYER CODE OF SHAPE', self.image_code_target_layer.shape, 'with min,max = %f,%f' % (self.image_code_target_layer.min(), self.image_code_target_layer.max())

            if run_back:
                diffs = self.net.blobs[backprop_layer].diff * 0
                diffs[0][backprop_unit] = self.net.blobs[backprop_layer].data[0,backprop_unit]

                assert back_mode in ('grad', 'deconv')
                if back_mode == 'grad':
                    with WithTimer('CaffeProcThread:backward', quiet = self.debug_level < 1):
                        #print '**** Doing backprop with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
                        try:
                            self.net.backward_from_layer(backprop_layer, diffs, zero_higher = True)
                        except AttributeError:
                            print 'ERROR: required bindings (backward_from_layer) not found! Try using the deconv-deep-vis-toolbox branch as described here: https://github.com/yosinski/deep-visualization-toolbox'
                            raise
                else:
                    with WithTimer('CaffeProcThread:deconv', quiet = self.debug_level < 1):
                        #print '**** Doing deconv with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
                        try:
                            self.net.deconv_from_layer(backprop_layer, diffs, zero_higher = True)
                        except AttributeError:
                            print 'ERROR: required bindings (deconv_from_layer) not found! Try using the deconv-deep-vis-toolbox branch as described here: https://github.com/yosinski/deep-visualization-toolbox'
                            raise

                with self.state.lock:
                    self.state.back_stale = False

            if run_upconv:
                # Much copied from https://github.com/Evolving-AI-Lab/synthesizing/blob/master/act_max.py

                if self.upconv_code is None:
                    self.upconv_code_shape = self.upconv_net.blobs[self.upconv_in_layer].data.shape
                    self.upconv_code = np.random.normal(0, .1, self.upconv_code_shape)
                    self.upconv_code = np.maximum(np.minimum(self.upconv_code, self.upconv_code_clip), 0)
                    print 'Initial code with shape', self.upconv_code.shape

                self.upconv_net.forward(feat=self.upconv_code)
                self.im_upconv_blob = self.upconv_net.blobs[self.upconv_out_layer].data


                #pdb.set_trace()
                
                #print 'self.im_upconv_blob shape is', self.im_upconv_blob.shape

                # Crop from 256x256 to 227x227
                ##print '   REMOVE COPY HERE'
                self.im_upconv_crop_blob = self.im_upconv_blob.copy()[:,:,self.topleft[0]:self.topleft[0]+self.input_dims[0], self.topleft[1]:self.topleft[1]+self.input_dims[1]]
                #print 'self.im_upconv_crop_blob shape is', self.im_upconv_crop_blob.shape
                
                # 2. forward pass the image self.im_upconv_crop_blob to net to maximize an unit k
                # 3. backprop the gradient from net to the image to get an updated image x

                with WithTimer('CaffeProcThread:forward upconv im', quiet = self.debug_level < 1):
                    self.net.forward(data=self.im_upconv_crop_blob)
                    #net_preproc_forward(self.net, self.im_upconv_crop_blob, self.input_dims)

                n_elem = np.prod(self.image_code_target_layer.shape)
                normalize_by = n_elem
                normalize_by = 1.0
                #normalize_by = np.sqrt(n_elem)

                # 4. Compute diffs for cost C = .5 * ()^2 w.r.t. upconv code
                # C = .5 * (((self.net.blobs[layer].data - self.image_code_target_layer)/normalize_by)**2).sum()

                upconv_code_target_layer = self.net.blobs[layer].data


                #####MASK

                #code_diff = upconv_code_target_layer - self.image_code_target_layer
                #SUPER HACK
                #code_diff -= 10
                code_diff = -self.image_code_target_layer.copy()
                #code_diff += -code_diff.max() - 1
                if self.state.cursor_area == 'top':
                    top_k = (-self.image_code_target_layer.flatten()).argsort()
                    mask = np.zeros_like(self.image_code_target_layer)
                    mask.ravel()[top_k[0:10]] = 1.0
                else:
                    mask = np.zeros_like(code_diff)
                    mask[0,self.state.selected_unit] = 1.0
                    # hack to manually control fc layers
                    if mask.sum() == 1:
                        code_diff[0,self.state.selected_unit] = -1.0       # corner case: negative pre-softmax activations
                code_diff *= mask

                try:
                    cost = .5 * (((code_diff) / normalize_by)**2).sum()
                except:
                    print 'ERROR'
                    pdb.set_trace()
                #print 'upconv code match cost is', cost
                diffs = (code_diff) / normalize_by**2
                #diffs = np.zeros((1,1000), 'float32')
                #diffs[0,717] = -1.0
                #diffs[0,933] = -1.0
                print diffs
                print diffs.min(), diffs.max()
                #print 'Truck image code is %g, upconv code is %g' % (self.image_code_target_layer[0,717], upconv_code_target_layer[0,717])
                #pdb.set_trace()
                
                
                with WithTimer('CaffeProcThread:backward upconv im', quiet = self.debug_level < 1):
                    #print '**** Doing backprop with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
                    self.net.backward_from_layer(layer, diffs, zero_higher = True)
                    #self.net.backward_from_layer('fc8', diffs, zero_higher = True)

                grad_blob = self.net.blobs['data'].diff

                # ADD grad of image matching cost
                #grad_blob += .000001 * (self.net.blobs['data'].data - self.image_input)
                #grad_blob = .000001 * (self.net.blobs['data'].data - self.image_input)
                
                # Manually deprocess (skip mean subtraction and rescaling)
                #grad_img = self.net.deprocess('data', diff_blob)
                #grad_blob = grad_blob[0]                    # bc01 -> c01
                #grad_blob = grad_blob.transpose((1,2,0))    # c01 -> 01c
                #grad_img = grad_blob[:, :, self._net_channel_swap_inv]  # e.g. BGR -> RGB
                #print 'CHECK ON RGB / BRG order'

                # Push back through upconv net to get grad dC / dcode
                self.upconv_image_grad_blob[:,:,self.topleft[0]:self.topleft[0]+self.input_dims[0], self.topleft[1]:self.topleft[1]+self.input_dims[1]] = grad_blob
                self.upconv_net.backward_from_layer(self.upconv_out_layer, self.upconv_image_grad_blob)
                grad_code = self.upconv_net.blobs[self.upconv_in_layer].diff
                grad_code /= (np.linalg.norm(grad_code) + 1e-6)
                #print 'got grad_code with min,max = %f,%f' % (grad_code.min(), grad_code.max())

                desired_prog = cost / 20
                max_lr = 1e6
                upconv_prog_lr = desired_prog / np.linalg.norm(grad_code)**2
                upconv_lr = min(max_lr, upconv_prog_lr)

                upconv_lr = 10.0   # Works everywhere??
                #upconv_lr = .0001   # conv1?
                #upconv_lr = .0001   # ok for conv5
                #upconv_lr = .1     # conv5 mask10??
                #upconv_lr = .01     # conv5 mask channel??
                #upconv_lr = .001     # conv5 nomask??
                #upconv_lr = 2.0     # ok for fc8
                noise_lr = .01
                noise_lr = 0.1
                code_decay = .99
                #code_decay = 1.0
                print 'upconv_lr =', upconv_lr
                
                self.upconv_code += -upconv_lr * grad_code + noise_lr * np.random.normal(0, 1, self.upconv_code_shape)
                #self.upconv_code = np.maximum(np.minimum(self.upconv_code, self.upconv_code_clip), 0)
                self.upconv_code = np.maximum(np.minimum(self.upconv_code, self.upconv_code_clip.max()), 0)
                self.upconv_code *= code_decay

                if self.state.upconv_pokes > 0:
                    self.upconv_code += np.random.normal(0, 2*self.state.upconv_pokes, self.upconv_code_shape)
                    self.state.upconv_pokes = 0
                elif self.state.upconv_pokes == -1:
                    self.upconv_code = np.random.normal(0, .1, self.upconv_code_shape)
                    self.upconv_code = np.maximum(np.minimum(self.upconv_code, self.upconv_code_clip), 0)
                    self.state.upconv_pokes = 0
                    
                    
                    
                print ('Im target code %g,%g,%g    upconv target code %g,%g,%g,   cost %g,  lr %g' %
                       (self.image_code_target_layer.min(), self.image_code_target_layer.max(), np.linalg.norm(self.image_code_target_layer),
                        upconv_code_target_layer.min(), upconv_code_target_layer.max(), np.linalg.norm(upconv_code_target_layer),
                        #self.upconv_code.min(), self.upconv_code.max(), np.linalg.norm(self.upconv_code),
                        cost, upconv_lr)) 
                #print 'self.upconv_code min,max = %f,%f' % (self.upconv_code.min(), self.upconv_code.max())
                ## OLD
                ## 4. Place the changes in x (227x227) back to self.im_upconv_crop_blob (256x256)
                #updated_self.im_upconv_crop_blob = self.im_upconv_crop_blob.copy()
                #updated_self.im_upconv_crop_blob[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = x.copy()
                #
                ## 5. backprop the image to generator to get an updated code
                #grad_norm_generator, updated_code = make_step_generator(net=generator, x=updated_self.im_upconv_crop_blob, self.im_upconv_crop_blob=self.im_upconv_crop_blob,
                #    start=self.upconv_in_layer, end=self.upconv_out_layer, step_size=step_size)

                #pdb.set_trace()
                

            if run_fwd or run_back or run_upconv:
                with self.state.lock:
                    self.state.caffe_net_state = 'free'
                    self.state.drawing_stale = True
                    #print '============ marking as stale (and sleeping)'
                now = time.time()
                if self.last_process_finished_at:
                    self.last_process_elapsed = now - self.last_process_finished_at
                self.last_process_finished_at = now
                time.sleep(self.loop_sleep)
            else:
                #######print '============ not marking as stale, just sleeping'
                time.sleep(self.loop_sleep)
        
        print 'CaffeProcThread.run: finished'
        print 'CaffeProcThread.run: processed %d frames fwd, %d frames back' % (self.frames_processed_fwd, self.frames_processed_back)

    def approx_fps(self):
        '''Get the approximate frames per second processed by this
        thread, considering only the last image processed. If more
        than two seconds ago, assume pipeline has stalled elsewhere
        (perhaps using static images that are only processed once).
        '''
        if self.last_process_elapsed and (time.time() - self.last_process_finished_at) < 2.0:
            return 1.0 / (self.last_process_elapsed + 1e-6)
        else:
            return 0.0
