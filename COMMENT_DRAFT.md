@paulbkoch That makes a lot of sense — separating progress callbacks from feature-examination callbacks is a cleaner design. 

Before I start on this, I have a few questions to make sure I get the design exactly right:

1. **Keyword argument names**: What exact parameter names do you want for each callback type? 
   - For example, for the progress callback: `def progress_cb(*, bag_idx, n_steps, best_score):`? 
   - And for the examination callback: `def exam_cb(*, bag_idx, n_steps, term_idx, avg_gain):`? Or different names?

2. **Backward compatibility**: The current callback uses positional args `(bag_idx, n_steps, has_progressed, best_score)`. Should we keep supporting the old positional-arg style during a deprecation period, or is a clean break acceptable?

3. **Tuple support now or later?**: You mentioned eventually passing tuples like `callback=(progress_cb, exam_cb)`. Should I implement tuple support in this PR, or just the keyword-arg introspection for now and leave tuples for a follow-up?
