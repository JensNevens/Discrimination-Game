;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; The Discrimination Game ;;
;;       Jens Nevens       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;
;; Global Parameters ;;
;;;;;;;;;;;;;;;;;;;;;;;
(defparameter *DATA* '())
(defparameter *QUEUE-SIZE* 25)

;; Global parameters that can be used to alter behavior
(defparameter *SALIENCY-TRESHOLD* 0)

(defparameter *PRUNING* nil) ;;nil = no pruning, T = pruning
(defparameter *AGE-TRESH* 150)
(defparameter *SUCCESS-TRESH* 0.075)
(defparameter *PRUNING-FREQ* 0)
(defparameter *PRUNING-COUNT* *PRUNING-FREQ*)

(defparameter *CHANNEL-CRITERIA* 0) ;; 0=success or 1=simplicity

;;;;;;;;;;;;;;;;
;; Game Queue ;;
;;;;;;;;;;;;;;;;
;; The game queue remembers the outcome of the N previous games
;; This is used to compute a running average of the success rate.
;; The queue has a fixed size, which can be manipulated through *QUEUE-SIZE*
;; The queue is implemented as a circular vector

(defstruct queue
  (elements (make-array *QUEUE-SIZE* :initial-element nil))    
  (put-ptr 0)
  (get-ptr 0))

(defun enqueue (elm q)
	(if (= (mod (+ 1 (queue-put-ptr q)) *QUEUE-SIZE*) (queue-get-ptr q))
		(progn
			(dequeue q)
			(enqueue elm q))
		(progn
			(setf (elt (queue-elements q) (queue-put-ptr q)) elm)
			(setf (queue-put-ptr q) (mod (+ (queue-put-ptr q) 1) *QUEUE-SIZE*)))))

(defun dequeue (q)
	(if (= (queue-get-ptr q) (queue-put-ptr q))
		nil
		(let ((elm (elt (queue-elements q) (queue-get-ptr q))))
			(setf (queue-get-ptr q) (mod (- (queue-get-ptr q) 1)*QUEUE-SIZE*))
			elm)))

(defun log-game (outcome q)
	(enqueue outcome q))


;;;;;;;;;;;
;; Input ;;
;;;;;;;;;;;
;; Read the input file, parse it, get the right sensory data
;; and store in *DATA*

(defun read-file ()
	(let ((in (open "./qrio-1/object-features.txt")))
		(read-line in)
		(when in
			(loop for line = (read-line in nil)
				  while line do (setq *DATA* (cons (select-channels (my-split line)) *DATA*)))
			(close in))))

(defun my-split (string &key (delimiterp #'delimiterp))
  (loop :for beg = (position-if-not delimiterp string)
    :then (position-if-not delimiterp string :start (1+ end))
    :for end = (and beg (position-if delimiterp string :start beg))
    :when beg :collect (read-from-string (string-trim " " (subseq string beg end)))
    :while end))

(defun delimiterp (c) (char= c #\,))

(defun select-channels (lst)
	(let ((channels '(0 1 3 4 5)))
		(loop for i in channels
			  collect (nth i lst))))

;; Select some objects from the data (the scene)
;; Select one object from the data (the topic)
(defun select-topic ()
	(let ((sze (list-length *DATA*)))
		(nth (random sze) *DATA*)))

(defun select-scene ()
	(let ((sze (list-length *DATA*))
		  (scene-sze 4)
		  (scenes '()))
		(dotimes (i scene-sze)
			(setq scenes (cons (nth (random sze) *DATA*) scenes)))
		scenes))

;;;;;;;;;;
;; Node ;;
;;;;;;;;;;
;; Each node of a Discrimination Tree has a range
;; a use-counter, success-counter, age-counter,
;; a left child and a right child
(defstruct node
	(min     0)
	(max     1)
	(used    0)
	(success 0)
	(age     0)
	(left    nil)
	(right   nil))

;;;;;;;;;;;
;; Agent ;;
;;;;;;;;;;;
;; An agent keeps track of its discrimination trees
(defstruct agent
	(trees (loop repeat 5 collect (make-node))))

;;;;;;;;;;;;;;;;;;;
;; Preprocessing ;;
;;;;;;;;;;;;;;;;;;;
;; The topic and scene objects are preproccessed
;; Saliency is computed and context scaling is performed
;; When saliency of a channel is below *SALIENCY-TRESHOLD*
;; this channel will not be considered during categorisation.
;; Context scaling is performed on all channels, except the grayscale channel

(defun saliency (topic scenes)
	(let ((result (list nil nil nil nil nil)))
		(loop for scene in scenes
			  do (loop for i from 0 to (- (list-length topic) 1)
			  	       for sce = (nth i scene)
			  	       for top = (nth i topic)
			  	       for salien = (abs (- sce top))
			  	       when (null (nth i result))
			  	       	do (setf (nth i result) salien)
			  	       when (< salien (nth i result))
			  	       	do (setf (nth i result) salien))
			  finally (return result))))

;; Context scaling
(defun context-scaling (topic scenes)
	(let ((topicc (copy-tree topic))
		  (scenesc (copy-tree scenes)))
		(loop for i from 0 to (- (list-length topic) 2)
			  for top-val = (nth i topic)
			  for scene-vals = (mapcar (lambda (x) (nth i x)) scenes)
			  do (multiple-value-bind (tp sc) (scale top-val scene-vals)
			  		(progn 
			  			(setf (nth i topicc) tp)
			  			(loop for j from 0 to (- (list-length sc) 1)
			  	       		  do (setf (nth i (nth j scenesc)) (nth j sc)))))
		      finally (return (values topicc scenesc)))))

(defun scale (topic scene)
	(let* ((minm (min topic (apply #'min scene)))
		   (maxm (max topic (apply #'max scene)))
		   (top (/ (- topic minm) (- maxm minm))))
		(loop for val in scene
			  collect (/ (- val minm) (- maxm minm)) into scaled-scene
			  finally (return (values top scaled-scene)))))

(defun preprocess (topic scenes)
	(let ((salien (saliency topic scenes))
		  (consider-channels (list 0 1 2 3 4)))
		(loop for i from 0 to (- (list-length salien) 1)
			  when (< (nth i salien) *SALIENCY-TRESHOLD*)
			  	do (setf consider-channels (remove i consider-channels))
		      finally 
		      	(return 
		      		(multiple-value-bind (scaled-topic scaled-scenes) (context-scaling topic scenes)
		      			(values consider-channels scaled-topic scaled-scenes))))))

;;;;;;;;;;;;;;;;;;;;
;; Categorisation ;;
;;;;;;;;;;;;;;;;;;;;
;; For categorisation, the powerset of the categorisers is computed
;; Each element of the powerset is tried. When categorisation
;; succeeds with given element, the path of the topic through the
;; discrimination trees is returned. All possible solutions are
;; returned. The best of these will be computed later on.
(defun combinations (lst)
	(let ((powerset (subsets lst)))
		(sort powerset #'(lambda (x y) (< (list-length x) (list-length y))))
		(remove-if (lambda (x) (< (list-length x) 1)) powerset)))

(defun subsets (lst)
  (if (null lst)
      '(nil)
      (let ((sub (subsets (cdr lst))))
      	(append sub (mapcar (lambda (x) (cons (car lst) x)) sub)))))

(defun categorise (agent topic scenes channels)
	(let ((trees (agent-trees agent))
		  (combos (combinations (list 0 1 2 3 4)))
		  (ctr 0)
		  (solutions '())
		  (thisSolution '()))
		(loop for combo in combos
			  do (setf ctr (list-length combo))
			  do (loop for i in combo
			  	       for tree = (nth i trees)
			  	       for topic-val = (nth i topic)
			  	       for scene-vals = (mapcar (lambda (x) (nth i x)) scenes)
			  	       do (let ((node (tree-run tree topic-val scene-vals '())))
			  	       		(when (and path (member i channels))
			  	       			  (progn
			  	       				(setf ctr (- ctr 1))
			  	       				(setf thisSolution (cons node thisSolution))))))
			  if (= ctr 0)
			  	do (progn
			  		(setf channels (cons thisChannel channels))
			  		(setf paths (cons thisPath paths))
			  		(setf thisChannel '())
			  		(setf thisPath '())
			  		(setf ctr 0))
			  else
			  	do (progn
			  		(setf thisChannel '())
			  		(setf thisPath '())
			  		(setf ctr 0))
			  finally (return (values channels paths)))))

(defun tree-run (tree topic-val scene-vals path)
	(let* ((min (node-min tree))
		   (max (node-max tree))
		   (half (abs (float (/ (+ max min) 2)))))
		(if (not (null scene-vals))
			(cond
				((and (< topic-val half) (>= topic-val min) (not (null (node-left tree))))
					(progn
						(setf scene-vals (remove-if (lambda (x) (or (>= x half) (< x min))) scene-vals))
						(tree-run (node-left tree) topic-val scene-vals (cons tree path))))
				((and (>= topic-val half) (<= topic-val max) (not (null (node-right tree))))
					(progn
						(setf scene-vals (remove-if (lambda (x) (or (< x half) (> x max))) scene-vals))
						(tree-run (node-right tree) topic-val scene-vals (cons tree path))))
				(t nil))
			(progn
				(setf path (cons tree path))
				path))))

;;;;;;;;;;;;;;
;; Counters ;;
;;;;;;;;;;;;;;
;; Counters of the nodes need to be updated. The age counter is incremented
;; after every game. The success counter is updated by using the path of the
;; discriminator. The use counter is incremened on every node the topic
;; runs through.
(defun update-age (agent)
	(let ((trees (agent-trees agent)))
		(loop for tree in trees
			  do (tree-age tree))))

(defun tree-age (tree)
	(progn
		(setf (node-age tree) (+ (node-age tree) 1))
		(when (and (not (null (node-left tree))) (not (null (node-right tree))))
			(progn
				(tree-age (node-left tree))
				(tree-age (node-right tree))))))

(defun update-success (path-list)
	(loop for possibility in path-list
	      do (loop for path in possibility
	      	       do (loop for node in path
	      	       	        do (setf (node-success node) (+ (node-success node) 1))))))

(defun update-use (agent topic scenes consider)
	(let ((trees (agent-trees agent)))
		(loop for tree in trees
			  for i from 0 to (- (list-length trees) 1)
			  for topic-val = (nth i topic)
			  for scene-vals  = (mapcar (lambda (x) (nth i x)) scenes)
			  when (member i consider)
			  	do (go-down tree topic-val scene-vals))))

(defun go-down (tree topic-val scene-vals)
	(let* ((min (node-min tree))
		   (max (node-max tree))
		   (half (abs (float (/ (+ max min) 2)))))
		(if (not (null scene-vals))
			(cond
				((and (< topic-val half) (>= topic-val min) (not (null (node-left tree))))
					(progn
						(setf scene-vals (remove-if (lambda (x) (or (>= x half) (< x min))) scene-vals))
						(setf (node-used tree) (+ (node-used tree) 1))
						(go-down (node-left tree) topic-val scene-vals)))
				((and (>= topic-val half) (<= topic-val max) (not (null (node-right tree))))
					(progn
						(setf scene-vals (remove-if (lambda (x) (or (< x half) (> x max))) scene-vals))
						(setf (node-used tree) (+ (node-used tree) 1))
						(go-down (node-right tree) topic-val scene-vals)))
				(t 
					(setf (node-used tree) (+ (node-used tree) 1))))
			(setf (node-used tree) (+ (node-used tree) 1)))))

;;;;;;;;;;;;
;; Growth ;;
;;;;;;;;;;;;
;; A node is randomly chosen and expanded
(defun growth (agent)
	(let ((channel (random 5)))
		(grow (nth channel (agent-trees agent)))))

(defun grow (node)
	(if (and (null (node-left node)) (null (node-right node)))
		(let* ((min (node-min node))
		       (max (node-max node))
		       (half (abs (float (/ (+ max min) 2))))
		       (left (make-node :min min :max half))
		       (right (make-node :min half :max max)))
			(setf (node-left node) left)
			(setf (node-right node) right)
			node)
		(let ((direction (random 2)))
			(case direction
				(0 (grow (node-left node)))
			    (1 (grow (node-right node)))))))

;;;;;;;;;;;;;
;; Pruning ;;
;;;;;;;;;;;;;
;; Pruning can be turned of by putting *PRUNING* to nil.
;; Pruning happens every N games. N can be chosen by putting *PRUNING-FREQ* to a
;; certain number. When pruning takes place, every node of every tree is checked
;; to see if pruning can be applied. This is true when the age of the node is above
;; *AGE-TRESH* and the success rate of the node is below *SUCCESS-RATE*.
(defun prune? (agent)
	(when *PRUNING*
		(if (= *PRUNING-COUNT* 0)
			(progn
				(pruning agent)
				(setf *PRUNING-COUNT* *PRUNING-FREQ*))
			(setf *PRUNING-COUNT* (- *PRUNING-COUNT* 1)))))

(defun pruning (agent)
	(let ((trees (agent-trees agent)))
		(loop for tree in trees
			  do (prune tree))))

(defun prune (curr)
	(cond
		((and (null (node-left curr)) (null (node-right curr)))
			nil)
		((and (can-prune (node-left curr)) (can-prune (node-right curr)))
			(progn
				(setf (node-left curr) nil)
				(setf (node-right curr) nil)))
		(t
			(progn
				(prune (node-left curr))
				(prune (node-right curr))))))

(defun can-prune (node)
	(let ((age (node-age node))
		  (use (node-used node))
		  (success (node-success node)))
		(if (not (= use 0))
			(if (and (not (null (node-left node))) (not (null (node-right node))))
				(and (> age *AGE-TRESH*)
					 (< (float (/ success use)) *SUCCESS-TRESH*)
					 (can-prune (node-left node))
					 (can-prune (node-right node)))
				(and (> age *AGE-TRESH*)
					 (< (float (/ success use)) *SUCCESS-TRESH*)))
			nil)))

;;;;;;;;;;;;;;;;;
;; Choose best ;;
;;;;;;;;;;;;;;;;;
;; When multiple solutions are found for categorisation, the best one needs to be chosen
;; This can be done based on success in earlier games or based on simplicity.
(defun choose-best (paths)
	(let ((best-path (case *CHANNEL-CRITERIA*
						(0 (choose-success paths))
						(1 (choose-simple paths)))))
		best-path))

(defun choose-success (paths)
	(let ((best-path 0)
		  (best-success -1))
		(loop for p in paths
			  for i from 0 to (- (list-length paths) 1)
			  for j = (list-length p)
			  if (> j 1)
			  	do (loop for x in p
			  		     for nde = (car x)
			  		     collect (node-success nde) into s
			  		     do (let ((res (/ (apply #'+ s) j)))
			  		     		(when (> res best-success)
			  		     			(progn
			  		     				(setf best-success res)
			  		     				(setf best-path (nth i paths))))))
			  else
			  	do (loop for x in p
			  	         for nde = (car x)
			  	         for success = (node-success nde)
			  	         when (> success best-success)
			  	       	  do (progn
			  	       			(setf best-success success)
			  	       			(setf best-path (nth i paths))))
			  finally (return best-path))))

(defun choose-simple (paths)
	(let ((best-path 0)
		  (simplest 100))
		(loop for p in paths
			  for i from 0 to (- (list-length paths) 1)
			  for l = (list-length p)
			  when (< l simplest)
			  	do (progn
			  			(setf simplest l)
			  			(setf best-path (nth i paths))))))

;;;;;;;;;;;;;;;;
;; Statistics ;;
;;;;;;;;;;;;;;;;
;; Statistics are gathered and written to a .csv file
;; The running average of success rate is gatehred, together with
;; the repertoize size at each game.
(defun gather-stats (agent game-queue)
	(let ((rep-sze (repertoire-size agent))
		  (suc-rte (success-rate game-queue)))
		(cons rep-sze suc-rte)))

(defun repertoire-size (agent)
	(let ((trees (agent-trees agent)))
		(loop for tree in trees
			  summing (tree-size tree))))

(defun tree-size (tree)
	(if (and (null (node-left tree)) (null (node-right tree)))
		1
		(+ 1
		   (tree-size (node-left tree))
		   (tree-size (node-right tree)))))

(defun success-rate (game-queue)
	(let* ((elements (queue-elements game-queue))
		   (success (count 'S elements)))
		(when (not (find nil elements))
			  (* (float (/ success (length elements))) 100))))

(defun write-data (data)
	(let ((stream (open "./data.csv" :direction :output :if-exists :supersede)))
		(write-line "S,R" stream)
		(loop for pair in data
			  when pair
			  	do (write-line (format nil "~A, ~A" (car pair) (cdr pair)) stream)
			  finally (close stream))))


;;;;;;;;;;;;;;;
;; MAIN LOOP ;;
;;;;;;;;;;;;;;;
(defparameter *AGENT* (make-agent))

(defun play-n-games (n)
	(let ((game-queue (make-queue))
		  (stats '()))
		(read-file)
		(dotimes (i n)
			(let ((*TOPIC* (select-topic))
				  (*SCENE* (select-scene)))
				(multiple-value-bind (consider-channels scaled-topic scaled-scenes) (preprocess *TOPIC* *SCENE*)
					(multiple-value-bind (channels paths) (categorise *AGENT* scaled-topic scaled-scenes consider-channels)
						(progn
							; (pprint scaled-topic)
							; (pprint scaled-scenes)
							; (pprint *AGENT*)
							; (pprint channels)
							; (pprint paths)
							(let ((x (list-length channels)))
								(update-age *AGENT*)
								(update-use *AGENT* scaled-topic scaled-scenes consider-channels)
								(case x
									(0 (progn
											(log-game 'F game-queue)
											(growth *AGENT*)
											(print "F")))
									(1 (progn
											(log-game 'S game-queue)
											(update-success paths)
											(prune? *AGENT*)
											(print "S")))
									(t 
										(let ((best-path (choose-best paths)))
											(update-success (list best-path))
											(log-game 'S game-queue)
											(prune? *AGENT*)
											(print "S"))))
								(setf stats (cons (gather-stats *AGENT* game-queue) stats))))))))
		(write-data (reverse stats))))

(play-n-games 1500)

