from functools import wraps
import inspect

def require_length(n: int, *target_var_names: str):
    def decorator(func):
        sig = inspect.signature(func) # Get function signature
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs) # bing arguments to their name
            bound_args.apply_defaults() # apply default values
            
            for name in target_var_names: # check length
                value = bound_args.arguments[name]
                if name not in bound_args.arguments: # check if specified argument exists
                    raise ValueError(f"Decorator targets parameter '{name}', but '{func.__name__}' does not have this parameter.")
                if value is not None: # Only check if not None. None can be handled within the functions
                    if not hasattr(value, '__len__'): # Check if length is applicable
                            raise TypeError(f"Argument '{name}' is of type {type(value)} and has no length.")
                    if len(value) != n: # 
                        raise ValueError(f"Argument '{name}' must have length {n}, got {len(bound_args.arguments[name])}")

            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_bounds(lower_bound: float, upper_bound: float, *target_var_names: str):
    def decorator(func):
        sig = inspect.signature(func) # Get function signature
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs) # bing arguments to their name
            bound_args.apply_defaults() # apply default values
            
            for name in target_var_names: # check length
                value = bound_args.arguments[name]
                if name not in bound_args.arguments: # check if specified argument exists
                    raise ValueError(f"Decorator targets parameter '{name}', but '{func.__name__}' does not have this parameter.")
                if value is not None: # Only check if not None. None can be handled within the functions
                    if not hasattr(value, '__gt__') and not hasattr(value, '__lt__'): # Check if length is applicable
                            raise TypeError(f"Argument '{name}' is of type {type(value)} and has no < and > comparison.")
                    if value < lower_bound or value > upper_bound: # 
                        raise ValueError(f"Argument '{name}' must be between {lower_bound} and {upper_bound}, got {value}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
