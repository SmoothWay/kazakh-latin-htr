FROM golang:1.17.1-alpine as builder
WORKDIR /build
COPY go.mod .
RUN go mod download
COPY . .
RUN go build -o /main ./cmd/api
FROM alpine:3
COPY --from=builder main /bin/main
# ENV VIRTUAL_HOST=
EXPOSE 8080
ENTRYPOINT ["/bin/main"]