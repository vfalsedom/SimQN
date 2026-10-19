Logging and random generator
====================================

The logging module
-----------------------------------

SimQN uses Loguru to output logs and monitoring data, with a standard Python
``logging.Logger`` as a compatibility interface. Existing logging calls,
``setLevel()``, handlers, filters, and percent-style message arguments continue
to work. The logging helpers can include the simulator's current time:

.. code-block:: Python

    import qns.utils.log as log

    s = Simulator()

    # install the log to the simulator to get simulator's internal status
    log.install(s)

Users can set the logging level and record logs:

.. code-block:: Python

    import logging
    
    log.logger.setLevel(logging.INFO)
    a = 1

    log.debug("debug message")
    log.info("info message %d", a)
    log.warn("warn message %d", a + 1)
    log.error("error message")
    log.critical("critical message")

The default handler forwards records to the application's Loguru sinks. On
import, SimQN replaces Loguru's built-in stderr sink with this default format:

.. code-block:: Python

    _FORMAT = (
        "<yellow>{time:HH:mm:ss.SSS}</yellow> | "
        "<yellow>sim={extra[sim_time]}</yellow> | "
        "<level>{level}</level> "
        "<cyan>[{name}:{line}]</cyan> "
        "<level>{extra[display_message]}</level>"
    )

``{time}`` shows system time. ``sim`` shows the simulator's current time from
``log.install(s)`` in seconds with three decimal places, for example
``14:30:12.345 | sim=65.123s | INFO [example:42] message``. Before a simulator is
installed, it displays ``sim=—``. Simulation time updates as events run and does
not wrap after 24 hours. Native Loguru calls using this formatter display the
same two clocks. The original ``[s.tc]`` message prefix is retained internally
for existing handlers, but omitted from the default display to avoid repetition.

Use ``format_record`` when adding an application sink with this format. It
supplies the additional fields for both SimQN and native Loguru records:

.. code-block:: Python

    import sys
    from loguru import logger as loguru_logger
    from qns.utils.log import format_record

    loguru_logger.remove()
    loguru_logger.add(sys.stderr, format=format_record)

Colors are enabled automatically for terminals that support them. Custom
application sinks are preserved. If the application has already removed the
built-in sink, its configuration remains in control. To customize the output,
configure Loguru in your application. For example, to retain the previous
plain-message stdout format:

.. code-block:: Python

    import sys
    from loguru import logger as loguru_logger

    loguru_logger.remove()
    loguru_logger.add(sys.stdout, format="{message}", level=0)

Both ``log.logger`` and the Loguru sinks apply their own level filters. The
original Python ``LogRecord`` is available in Loguru's
``record["extra"]["logging_record"]``, including any custom ``extra`` fields.

Finally, SimQN provides ``monitor()`` for date output. ``sep`` sets the separator, the default separator is "," (like csv files). ``with_time`` is a boolean indicating whether add a column to record the simulator's current time.

.. code-block:: Python

    log.monitor(data1, data2, data3, with_time=True)
    # message: time,data1,data2,data3

With the ``Timer`` entity provided by SimQN, it is easy to print the network's status periodically.

.. code-block:: python

    from qns.simulator.simulator import Simulator
    from qns.entity.timer.timer import Timer
    import qns.utils.log as log

    s = Simulator(0, 10, 1000)
    log.install(s)

    # the trigger function is log network status
    def trigger_func():
        print("network status")

    # set the timer with a period
    t1 = Timer("t1", 0, 10, 0.5, trigger_func)
    t1.install(s)

    # run the simulation
    s.run()

Random generator's seed
--------------------------

SimQN uses ``random`` library from both Python standard library and ``numpy`` when measuring qubits or generating random topologies. To make the simulation results reproducible, SimQN provides ``set_seed()`` to fix the random generator's seed:

.. code-block:: python

    from qns.utils.random import set_seed

    set_seed(1641801012) # fix the random generator's seed

``get_randint(low, high)`` can generates an random integer in [low, high]; ``get_rand(low, high)`` can generate a float random number in [low, high); and ``get_choice(a)`` selects an random element in list ``a``.
