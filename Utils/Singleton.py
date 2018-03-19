class Singleton(type):
    """
    Singleton metaclass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Visitor(object):
    """
    This class is a base class for the visitor design pattern,
    a subclass can implement visit methods for specific types
    for example, given two classes A and B, a visitor need to create two functions
    visit_A and visit_B and dynamically the visit method will route to the correct visit method
    given a subject of type A or B
    """

    def visit(self, subject):
        method_name = 'visit_' + type(subject).__name__
        method = getattr(self, method_name)
        if method is None:
            method = self.generic_visit
        return method(subject)

    def generic_visit(self, subject):
        print('Class {0} does not have a visit function in {1}'.format(type(subject).__name__,
                                                                             type(self).__name__))