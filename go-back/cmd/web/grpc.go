package main

import (
	"fmt"
	"log"
	"net"

	"google.golang.org/grpc"
)

func (app *application) gRPCListen() {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", gRpcPort))
	if err != nil {
		log.Fatalf("Failed to listen for gRPC %v", err)
	}

	s := grpc.NewServer()

	log.Printf("gRPC server started on port %s", gRpcPort)

	if err := s.Serve(lis); err != nil {
		log.Fatalf("Failed to listen for gRPC %v", err)
	}
}
