#!/usr/bin/env python

from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
from cassandra.auth import PlainTextAuthProvider

########################################################################

### connection variables for on-prem Cassandra or DSE
# CONTACT_POINTS = ["121.184.96.123"]
PORT = 9042
# USERNAME = "cnscada"
# PASSWORD = "cnscada123!@#"

# DB Settings
CONN_TIMEOUT = 10

########################################################################

class cassandraConnect:

    def __init__(self, list_contact_points, username, password):
        """
        Initialize Cassandra Cluster and begin session by connecting
        """

        AUTH_PROVIDER = PlainTextAuthProvider(username=username, password=password)

        self.cluster = Cluster(
            contact_points = list_contact_points,
            port = PORT,
            connect_timeout = CONN_TIMEOUT,
            load_balancing_policy = DCAwareRoundRobinPolicy(local_dc='datacenter1'),
            protocol_version = 4,
            auth_provider = AUTH_PROVIDER
        )
        # self.session = self.cluster.connect(KEYSPACE)
        self.session = self.cluster.connect()
        print("Connected to Cassandra Successfully: {} {}".format(list_contact_points, PORT))

    def disconnect(self):
        """
        Disconnect from Cassandra
        """

        self.cluster.shutdown()
        self.session.shutdown()
        return (0)

########################################################################